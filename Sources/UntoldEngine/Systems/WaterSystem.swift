//
//  WaterSystem.swift
//  UntoldEngine
//
// Copyright (C) Untold Engine Studios
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Port of Evan Wallace's WebGL Water (https://github.com/evanw/webgl-water).
//
// `WaterRenderer` is a self-contained render feature: it owns its own
// simulation textures, procedural geometry, compute pipelines (the heightfield
// solver) and render pipelines (caustics + raytraced pool/sphere/water), and
// it builds its own render graph. When enabled, `buildGameModeGraph()` hands
// the whole frame over to `WaterRenderer.shared.buildGraph()`, which renders
// directly into the drawable. The deferred PBR pipeline is bypassed entirely.

import CShaderTypes
import MetalKit
import simd

/// A real-world mesh (e.g. an ARKit scene-reconstruction MeshAnchor) used to occlude
/// the virtual water on visionOS. Vertex/index buffers are referenced directly (no
/// copy); positions are read with the given stride/offset.
public struct WaterOcclusionMesh {
    public let vertexBuffer: MTLBuffer
    public let vertexOffset: Int
    public let vertexStride: Int
    public let indexBuffer: MTLBuffer
    public let indexOffset: Int
    public let indexCount: Int
    public let indexType: MTLIndexType
    public let transform: simd_float4x4   // world-from-mesh-local (anchor transform)

    public init(vertexBuffer: MTLBuffer, vertexOffset: Int, vertexStride: Int,
                indexBuffer: MTLBuffer, indexOffset: Int, indexCount: Int,
                indexType: MTLIndexType, transform: simd_float4x4) {
        self.vertexBuffer = vertexBuffer
        self.vertexOffset = vertexOffset
        self.vertexStride = vertexStride
        self.indexBuffer = indexBuffer
        self.indexOffset = indexOffset
        self.indexCount = indexCount
        self.indexType = indexType
        self.transform = transform
    }
}

public final class WaterRenderer: @unchecked Sendable {
    public static let shared = WaterRenderer()

    // Simulation resolution and caustic-map resolution (match the original demo).
    private let simSize = 256
    private let causticSize = 1024
    private let planeDetail = 200

    // Enable flag, checked by buildGameModeGraph().
    public private(set) var isEnabled = false
    public var isPaused = false

    // When set before the renderer is created, the engine skips building the heavy
    // deferred G-buffer/light pipelines (44 bytes of tile memory) that the water
    // feature never uses. This keeps the demo within the iOS simulator's 32-byte
    // color tile-storage limit.
    public var waterOnlyMode = false

    // visionOS path: render per-eye into the XR drawable using the engine-provided
    // per-eye view/projection, with the pool placed by `modelMatrix` (world-from-local).
    // Set before the renderer is created (affects pipeline depth/blend setup).
    public var xrMode = false
    public var modelMatrix = matrix_identity_float4x4

    private var initialized = false

    // MARK: Simulation state
    private var simCurrent: MTLTexture?      // holds the up-to-date heightfield after each step
    private var simOther: MTLTexture?
    private var causticsTexture: MTLTexture?
    private var sceneDepth: MTLTexture?

    // Demo-supplied art.
    private var tilesTexture: MTLTexture?
    private var skyTexture: MTLTexture?
    private var defaultTiles: MTLTexture?
    private var defaultSky: MTLTexture?

    // Scene parameters (world space).
    private var sphereCenter = simd_float3(-0.4, -0.75, 0.2)
    private var sphereRadius: Float = 0.25
    private var lastSimSphereCenter = simd_float3(-0.4, -0.75, 0.2)
    private var lightDirection = simd_normalize(simd_float3(2.0, 2.0, -1.0))

    private struct Drop { var center: simd_float2; var radius: Float; var strength: Float }
    private var pendingDrops: [Drop] = []

    // MARK: Geometry
    private var planeVertexBuffer: MTLBuffer?
    private var planeIndexBuffer: MTLBuffer?
    private var planeIndexCount = 0
    private var poolVertexBuffer: MTLBuffer?
    private var poolVertexCount = 0
    private var sphereVertexBuffer: MTLBuffer?
    private var sphereVertexCount = 0

    // MARK: Compute pipelines
    private var dropPipeline = ComputePipeline()
    private var updatePipeline = ComputePipeline()
    private var normalPipeline = ComputePipeline()
    private var spherePipeline = ComputePipeline()

    // MARK: Render pipelines
    private var causticsRender: RenderPipeline?
    private var poolRender: RenderPipeline?
    private var sphereRender: RenderPipeline?
    private var surfaceAboveRender: RenderPipeline?
    private var surfaceBelowRender: RenderPipeline?
    private var occlusionRender: RenderPipeline?   // depth-only, real-world occlusion (XR)

    // Real-world occlusion meshes (visionOS scene reconstruction), set each frame.
    private let occlusionLock = NSLock()
    private var occlusionMeshes: [WaterOcclusionMesh] = []

    private init() {}

    // MARK: - Public API

    /// Turns the water feature on/off. The first time it is enabled it lazily
    /// builds all GPU resources (the engine renderer must already exist).
    public func setEnabled(_ enabled: Bool) {
        if enabled { ensureInitialized() }
        isEnabled = enabled
    }

    public func setSphere(center: simd_float3, radius: Float) {
        sphereCenter = center
        sphereRadius = radius
    }

    public func setSphereCenter(_ center: simd_float3) {
        sphereCenter = center
    }

    public func getSphereCenter() -> simd_float3 { sphereCenter }
    public func getSphereRadius() -> Float { sphereRadius }

    public func setLightDirection(_ dir: simd_float3) {
        lightDirection = simd_normalize(dir)
    }

    /// Adds a ripple. `center` components are in the water plane's [-1, 1] domain.
    public func addDrop(center: simd_float2, radius: Float = 0.03, strength: Float = 0.01) {
        pendingDrops.append(Drop(center: center, radius: radius, strength: strength))
    }

    /// Seeds the surface with a handful of random ripples (as the demo does on launch).
    public func seedRipples(count: Int = 20) {
        for i in 0 ..< count {
            let x = Float.random(in: -1 ... 1)
            let z = Float.random(in: -1 ... 1)
            let strength: Float = (i % 2 == 0) ? 0.01 : -0.01
            addDrop(center: simd_float2(x, z), radius: 0.03, strength: strength)
        }
    }

    public func setTilesTexture(_ texture: MTLTexture) { tilesTexture = texture }
    public func setSkyTexture(_ texture: MTLTexture) { skyTexture = texture }

    /// Real-world occlusion meshes (visionOS). Rendered depth-only before the water so
    /// real surfaces (floor, furniture, a meshed person) occlude it.
    public func setOcclusionMeshes(_ meshes: [WaterOcclusionMesh]) {
        occlusionLock.withLock { occlusionMeshes = meshes }
    }

    /// Resets the heightfield to a flat, still surface.
    public func reset() {
        clearSimTextures()
        lastSimSphereCenter = sphereCenter
        pendingDrops.removeAll()
    }

    // MARK: - Initialization

    public func ensureInitialized() {
        guard !initialized else { return }
        guard renderInfo.device != nil, renderInfo.library != nil else {
            handleError(.metalDeviceNotFound)
            return
        }
        buildPipelines()
        buildTextures()
        buildGeometry()
        buildDefaultArt()
        clearSimTextures()
        initialized = true
    }

    private func buildPipelines() {
        let device = renderInfo.device!
        let library = renderInfo.library!

        CreateComputePipeline(into: &dropPipeline, device: device, library: library,
                              functionName: "waterDropKernel", pipelineName: "Water Drop")
        CreateComputePipeline(into: &updatePipeline, device: device, library: library,
                              functionName: "waterUpdateKernel", pipelineName: "Water Update")
        CreateComputePipeline(into: &normalPipeline, device: device, library: library,
                              functionName: "waterNormalKernel", pipelineName: "Water Normal")
        CreateComputePipeline(into: &spherePipeline, device: device, library: library,
                              functionName: "waterSphereKernel", pipelineName: "Water Sphere")

        let sceneColor = renderInfo.presentColorPixelFormat
        // In XR the depth comes from the layer (reverse-Z), and content must composite
        // over passthrough, so use the engine depth format, reverse-Z compare, and
        // premultiplied-alpha blending. On macOS/iOS the water owns the frame: its own
        // non-reverse-Z depth and opaque output.
        let depthFormat: MTLPixelFormat = xrMode ? renderInfo.depthPixelFormat : .depth32Float
        let depthCompare: MTLCompareFunction = .lessEqual          // flipped to greaterEqual under reverse-Z
        let reverseZ = xrMode
        let blend: PipelineBlendMode = xrMode ? .alphaPremultiplied : .none

        causticsRender = CreatePipeline(
            vertexShader: "vertexWaterCaustics", fragmentShader: "fragmentWaterCaustics",
            vertexDescriptor: nil, colorFormats: [.rgba16Float], depthFormat: .invalid,
            depthEnabled: false, reverseZCompatible: false, name: "Water Caustics Pipeline"
        )

        func sceneStage(_ vs: String, _ fs: String, _ name: String) -> RenderPipeline? {
            CreatePipeline(
                vertexShader: vs, fragmentShader: fs, vertexDescriptor: nil,
                colorFormats: [sceneColor], depthFormat: depthFormat,
                depthCompareFunction: depthCompare, depthEnabled: true,
                reverseZCompatible: reverseZ, blendMode: blend, name: name
            )
        }
        poolRender = sceneStage("vertexWaterPool", "fragmentWaterPool", "Water Pool Pipeline")
        sphereRender = sceneStage("vertexWaterSphere", "fragmentWaterSphere", "Water Sphere Pipeline")
        surfaceAboveRender = sceneStage("vertexWaterSurface", "fragmentWaterSurfaceAbove", "Water Surface Above Pipeline")
        surfaceBelowRender = sceneStage("vertexWaterSurface", "fragmentWaterSurfaceBelow", "Water Surface Below Pipeline")

        if xrMode {
            occlusionRender = buildOcclusionPipeline(colorFormat: sceneColor, depthFormat: depthFormat)
        }
    }

    /// Depth-only pipeline: writes depth (real-world surfaces) with no color, so the
    /// later water draws are occluded via the reverse-Z depth test.
    private func buildOcclusionPipeline(colorFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) -> RenderPipeline? {
        guard let library = renderInfo.library,
              let vfn = library.makeFunction(name: "vertexWaterOcclusion") else { return nil }
        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Water Occlusion Pipeline"
        desc.vertexFunction = vfn
        desc.fragmentFunction = nil
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.colorAttachments[0].writeMask = []        // depth only
        desc.depthAttachmentPixelFormat = depthFormat
        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = sceneDepthCompareFunction(.lessEqual, reverseZCompatible: true)
        dd.isDepthWriteEnabled = true
        do {
            let ps = try renderInfo.device.makeRenderPipelineState(descriptor: desc)
            let ds = renderInfo.device.makeDepthStencilState(descriptor: dd)
            return RenderPipeline(pipelineState: ps, depthState: ds, success: true, name: "Water Occlusion Pipeline")
        } catch {
            handleError(.pipelineStateCreationFailed, "Water Occlusion Pipeline")
            return nil
        }
    }

    private func buildTextures() {
        let device = renderInfo.device!
        simCurrent = createTexture(
            device: device, label: "Water Sim A", pixelFormat: .rgba32Float,
            width: simSize, height: simSize, usage: [.shaderRead, .shaderWrite],
            storageMode: .shared
        )
        simOther = createTexture(
            device: device, label: "Water Sim B", pixelFormat: .rgba32Float,
            width: simSize, height: simSize, usage: [.shaderRead, .shaderWrite],
            storageMode: .shared
        )
        causticsTexture = createTexture(
            device: device, label: "Water Caustics", pixelFormat: .rgba16Float,
            width: causticSize, height: causticSize, usage: [.shaderRead, .renderTarget],
            storageMode: .private
        )
    }

    private func ensureSceneDepth(width: Int, height: Int) {
        if let depth = sceneDepth, depth.width == width, depth.height == height { return }
        sceneDepth = createTexture(
            device: renderInfo.device, label: "Water Scene Depth", pixelFormat: .depth32Float,
            width: width, height: height, usage: [.renderTarget], storageMode: .private
        )
    }

    private func clearSimTextures() {
        guard let a = simCurrent, let b = simOther else { return }
        let bytesPerRow = simSize * MemoryLayout<simd_float4>.stride
        let zeros = [simd_float4](repeating: .zero, count: simSize * simSize)
        let region = MTLRegionMake2D(0, 0, simSize, simSize)
        zeros.withUnsafeBytes { raw in
            a.replace(region: region, mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: bytesPerRow)
            b.replace(region: region, mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: bytesPerRow)
        }
    }

    // MARK: - Geometry generation

    private func buildGeometry() {
        let device = renderInfo.device!

        // Water surface plane: (detail+1)^2 grid of (x, y, 0) vertices with x,y in [-1,1].
        let n = planeDetail + 1
        var planeVerts = [simd_float3]()
        planeVerts.reserveCapacity(n * n)
        for j in 0 ..< n {
            for i in 0 ..< n {
                let x = Float(i) / Float(planeDetail) * 2.0 - 1.0
                let y = Float(j) / Float(planeDetail) * 2.0 - 1.0
                planeVerts.append(simd_float3(x, y, 0.0))
            }
        }
        var planeIndices = [UInt32]()
        planeIndices.reserveCapacity(planeDetail * planeDetail * 6)
        for j in 0 ..< planeDetail {
            for i in 0 ..< planeDetail {
                let a = UInt32(j * n + i)
                let b = UInt32(j * n + i + 1)
                let c = UInt32((j + 1) * n + i)
                let d = UInt32((j + 1) * n + i + 1)
                planeIndices.append(contentsOf: [a, c, b, b, c, d])
            }
        }
        planeIndexCount = planeIndices.count
        planeVertexBuffer = device.makeBuffer(bytes: planeVerts,
                                              length: planeVerts.count * MemoryLayout<simd_float3>.stride,
                                              options: .storageModeShared)
        planeVertexBuffer?.label = "Water Plane Vertices"
        planeIndexBuffer = device.makeBuffer(bytes: planeIndices,
                                             length: planeIndices.count * MemoryLayout<UInt32>.stride,
                                             options: .storageModeShared)
        planeIndexBuffer?.label = "Water Plane Indices"

        // Pool: unit cube [-1,1]^3, open top. The vertex shader remaps y so the
        // cube's y=+1 face becomes the floor and the y=-1 face (the rim) is removed.
        let c = [
            simd_float3(-1, -1, -1), simd_float3(1, -1, -1), simd_float3(1, -1, 1), simd_float3(-1, -1, 1),
            simd_float3(-1, 1, -1), simd_float3(1, 1, -1), simd_float3(1, 1, 1), simd_float3(-1, 1, 1),
        ]
        func quad(_ a: Int, _ b: Int, _ cc: Int, _ d: Int) -> [simd_float3] {
            [c[a], c[b], c[cc], c[a], c[cc], c[d]]
        }
        var poolVerts = [simd_float3]()
        poolVerts += quad(4, 5, 6, 7) // y=+1 face (becomes the floor after remap)
        poolVerts += quad(0, 4, 7, 3) // x=-1 wall
        poolVerts += quad(1, 2, 6, 5) // x=+1 wall
        poolVerts += quad(0, 1, 5, 4) // z=-1 wall
        poolVerts += quad(3, 7, 6, 2) // z=+1 wall
        poolVertexCount = poolVerts.count
        poolVertexBuffer = device.makeBuffer(bytes: poolVerts,
                                             length: poolVerts.count * MemoryLayout<simd_float3>.stride,
                                             options: .storageModeShared)
        poolVertexBuffer?.label = "Water Pool Vertices"

        // Sphere: unit UV sphere, positions only.
        let stacks = 24, slices = 24
        var sphereVerts = [simd_float3]()
        func point(_ st: Int, _ sl: Int) -> simd_float3 {
            let v = Float(st) / Float(stacks)
            let u = Float(sl) / Float(slices)
            let phi = v * Float.pi
            let theta = u * 2.0 * Float.pi
            return simd_float3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta))
        }
        for st in 0 ..< stacks {
            for sl in 0 ..< slices {
                let p0 = point(st, sl)
                let p1 = point(st + 1, sl)
                let p2 = point(st + 1, sl + 1)
                let p3 = point(st, sl + 1)
                sphereVerts += [p0, p1, p2, p0, p2, p3]
            }
        }
        sphereVertexCount = sphereVerts.count
        sphereVertexBuffer = device.makeBuffer(bytes: sphereVerts,
                                               length: sphereVerts.count * MemoryLayout<simd_float3>.stride,
                                               options: .storageModeShared)
        sphereVertexBuffer?.label = "Water Sphere Vertices"
    }

    private func buildDefaultArt() {
        let device = renderInfo.device!
        // 1x1 white tiles fallback.
        let tdesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false
        )
        tdesc.usage = .shaderRead
        defaultTiles = device.makeTexture(descriptor: tdesc)
        var white: [UInt8] = [220, 220, 225, 255]
        defaultTiles?.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0,
                              withBytes: &white, bytesPerRow: 4)

        // 1x1 sky-blue cubemap fallback.
        let cdesc = MTLTextureDescriptor.textureCubeDescriptor(
            pixelFormat: .rgba8Unorm, size: 1, mipmapped: false
        )
        cdesc.usage = .shaderRead
        defaultSky = device.makeTexture(descriptor: cdesc)
        var sky: [UInt8] = [140, 190, 235, 255]
        for slice in 0 ..< 6 {
            defaultSky?.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, slice: slice,
                                withBytes: &sky, bytesPerRow: 4, bytesPerImage: 4)
        }
    }

    // MARK: - Render graph

    public func buildGraph() -> RenderGraphResult {
        var graph = [String: RenderPass]()

        let simPass = RenderPass(id: "waterSim", dependencies: [], execute: { [weak self] cb in
            self?.encodeSimulation(cb)
        })
        graph[simPass.id] = simPass

        let causticsPass = RenderPass(id: "waterCaustics", dependencies: [simPass.id], execute: { [weak self] cb in
            self?.encodeCaustics(cb)
        })
        graph[causticsPass.id] = causticsPass

        let scenePass = RenderPass(id: "waterScene", dependencies: [causticsPass.id], execute: { [weak self] cb in
            self?.encodeScene(cb)
        })
        graph[scenePass.id] = scenePass

        return (graph, scenePass.id)
    }

    // MARK: - Simulation encoding

    private func encodeSimulation(_ commandBuffer: MTLCommandBuffer) {
        guard initialized else { return }
        // In XR the graph runs once per eye; step the sim only on the first eye so it
        // advances once per frame.
        if xrMode, renderInfo.currentEye != 0 { return }

        for drop in pendingDrops {
            encodeDrop(commandBuffer, drop)
        }
        pendingDrops.removeAll()

        encodeMoveSphere(commandBuffer, old: lastSimSphereCenter, new: sphereCenter)
        lastSimSphereCenter = sphereCenter

        if !isPaused {
            encodeUpdate(commandBuffer)
            encodeUpdate(commandBuffer)
        }
        encodeNormals(commandBuffer)
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder) {
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        let groups = MTLSize(width: (simSize + 15) / 16, height: (simSize + 15) / 16, depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
        encoder.endEncoding()
    }

    private func swapSim() { swap(&simCurrent, &simOther) }

    private func encodeDrop(_ commandBuffer: MTLCommandBuffer, _ drop: Drop) {
        guard let state = dropPipeline.pipelineState, let src = simCurrent, let dst = simOther,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Water Drop"
        encoder.setComputePipelineState(state)
        encoder.setTexture(src, index: 0)
        encoder.setTexture(dst, index: 1)
        var center = drop.center
        var radius = drop.radius
        var strength = drop.strength
        encoder.setBytes(&center, length: MemoryLayout<simd_float2>.stride, index: Int(waterSimPassDropCenterIndex.rawValue))
        encoder.setBytes(&radius, length: MemoryLayout<Float>.stride, index: Int(waterSimPassDropRadiusIndex.rawValue))
        encoder.setBytes(&strength, length: MemoryLayout<Float>.stride, index: Int(waterSimPassDropStrengthIndex.rawValue))
        dispatch(encoder)
        swapSim()
    }

    private func encodeMoveSphere(_ commandBuffer: MTLCommandBuffer, old: simd_float3, new: simd_float3) {
        guard let state = spherePipeline.pipelineState, let src = simCurrent, let dst = simOther,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Water Move Sphere"
        encoder.setComputePipelineState(state)
        encoder.setTexture(src, index: 0)
        encoder.setTexture(dst, index: 1)
        var oldC = old
        var newC = new
        var radius = sphereRadius
        encoder.setBytes(&oldC, length: MemoryLayout<simd_float3>.stride, index: Int(waterSimPassOldCenterIndex.rawValue))
        encoder.setBytes(&newC, length: MemoryLayout<simd_float3>.stride, index: Int(waterSimPassNewCenterIndex.rawValue))
        encoder.setBytes(&radius, length: MemoryLayout<Float>.stride, index: Int(waterSimPassSphereRadiusIndex.rawValue))
        dispatch(encoder)
        swapSim()
    }

    private func encodeUpdate(_ commandBuffer: MTLCommandBuffer) {
        guard let state = updatePipeline.pipelineState, let src = simCurrent, let dst = simOther,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Water Update"
        encoder.setComputePipelineState(state)
        encoder.setTexture(src, index: 0)
        encoder.setTexture(dst, index: 1)
        dispatch(encoder)
        swapSim()
    }

    private func encodeNormals(_ commandBuffer: MTLCommandBuffer) {
        guard let state = normalPipeline.pipelineState, let src = simCurrent, let dst = simOther,
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Water Normals"
        encoder.setComputePipelineState(state)
        encoder.setTexture(src, index: 0)
        encoder.setTexture(dst, index: 1)
        dispatch(encoder)
        swapSim()
    }

    // MARK: - Uniforms

    private func currentUniforms() -> WaterSceneUniforms? {
        guard let camera = CameraSystem.shared.activeCamera,
              let cameraComponent = scene.get(component: CameraComponent.self, for: camera) else {
            return nil
        }
        let view = SceneRootTransform.shared.effectiveViewMatrix(cameraComponent.viewSpace)

        // Projection: XR uses the engine-provided per-eye (reverse-Z) projection; the
        // desktop path uses its own 45° perspective.
        let projection: simd_float4x4
        if xrMode {
            projection = renderInfo.perspectiveSpace
        } else {
            let aspect = renderInfo.viewPort.y > 0 ? renderInfo.viewPort.x / renderInfo.viewPort.y : 1.0
            projection = matrixPerspectiveRightHand(
                fovyRadians: 45.0 * Float.pi / 180.0, aspectRatio: aspect, nearZ: 0.01, farZ: 100.0
            )
        }

        // In XR the pool is placed by `modelMatrix` (world-from-local). The raytraced
        // shaders work in pool-local space, so transform the eye into local space
        // (the light is already defined in local space).
        let worldEye = view.inverse.columns.3
        let eyeLocal: simd_float4 = xrMode ? simd_mul(simd_inverse(modelMatrix), worldEye) : worldEye

        var u = WaterSceneUniforms()
        u.mvp = xrMode
            ? simd_mul(simd_mul(projection, view), modelMatrix)
            : simd_mul(projection, view)
        u.eye = simd_float3(eyeLocal.x, eyeLocal.y, eyeLocal.z)
        u.light = lightDirection
        u.sphereCenter = sphereCenter
        u.sphereRadius = sphereRadius
        return u
    }

    // MARK: - Picking (uses the same camera view + projection as rendering)

    private func cameraViewProjection() -> (view: simd_float4x4, projection: simd_float4x4, eye: simd_float3)? {
        guard let camera = CameraSystem.shared.activeCamera,
              let cameraComponent = scene.get(component: CameraComponent.self, for: camera) else {
            return nil
        }
        let view = SceneRootTransform.shared.effectiveViewMatrix(cameraComponent.viewSpace)
        let aspect = renderInfo.viewPort.y > 0 ? renderInfo.viewPort.x / renderInfo.viewPort.y : 1.0
        let projection = matrixPerspectiveRightHand(
            fovyRadians: 45.0 * Float.pi / 180.0, aspectRatio: aspect, nearZ: 0.01, farZ: 100.0
        )
        let e = view.inverse.columns.3
        return (view, projection, simd_float3(e.x, e.y, e.z))
    }

    /// World-space ray (origin, normalized direction) for a screen point. `screenPoint`
    /// is in the same coordinate space as `viewport` with the origin at the TOP-left.
    public func screenRay(screenPoint: simd_float2, viewport: simd_float2) -> (origin: simd_float3, direction: simd_float3)? {
        guard let (view, projection, eye) = cameraViewProjection() else { return nil }
        let dir = rayDirectionInWorldSpace(
            uMouseLocation: screenPoint, uViewPortDim: viewport,
            uPerspectiveSpace: projection, uViewSpace: view
        )
        return (eye, simd_normalize(dir))
    }

    /// Where the ray through `screenPoint` meets the water plane (y=0), as an xz point
    /// in the water's [-1,1] domain — or nil if it misses the pool footprint.
    public func pickWaterPlane(screenPoint: simd_float2, viewport: simd_float2) -> simd_float2? {
        guard let (o, d) = screenRay(screenPoint: screenPoint, viewport: viewport) else { return nil }
        if abs(d.y) < 1e-5 { return nil }
        let t = -o.y / d.y
        if t <= 0 { return nil }
        let hit = o + d * t
        if abs(hit.x) > 1.0 || abs(hit.z) > 1.0 { return nil }
        return simd_float2(hit.x, hit.z)
    }

    /// The world-space point where the ray through `screenPoint` first hits the sphere,
    /// or nil if it misses. Used both as the sphere hit-test and the drag grab point.
    public func sphereHitPoint(screenPoint: simd_float2, viewport: simd_float2) -> simd_float3? {
        guard let (o, d) = screenRay(screenPoint: screenPoint, viewport: viewport) else { return nil }
        let oc = o - sphereCenter
        let b = simd_dot(oc, d)
        let c = simd_dot(oc, oc) - sphereRadius * sphereRadius
        let disc = b * b - c
        if disc <= 0 { return nil }
        let t = -b - sqrt(disc)
        if t <= 0 { return nil }
        return o + d * t
    }

    // MARK: - Caustics encoding

    private func encodeCaustics(_ commandBuffer: MTLCommandBuffer) {
        if xrMode, renderInfo.currentEye != 0 { return }  // caustics depend only on the sim
        guard initialized, let pipeline = causticsRender, pipeline.success,
              let causticsTexture, let water = simCurrent,
              let planeVertexBuffer, let planeIndexBuffer,
              var u = currentUniforms() else { return }

        let descriptor = MTLRenderPassDescriptor()
        descriptor.colorAttachments[0].texture = causticsTexture
        descriptor.colorAttachments[0].loadAction = .clear
        descriptor.colorAttachments[0].storeAction = .store
        // Clear to black: floor regions outside the caustic projection get no caustic
        // light (so the outer floor reads as shadow, matching the original) rather than
        // a uniform bright fill.
        descriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 0.0)

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        let uniformLength = MemoryLayout<WaterSceneUniforms>.stride
        encoder.label = "Water Caustics Pass"
        encoder.setRenderPipelineState(pipeline.pipelineState!)
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(planeVertexBuffer, offset: 0, index: Int(waterScenePassPositionIndex.rawValue))
        encoder.setVertexBytes(&u, length: uniformLength, index: Int(waterScenePassUniformIndex.rawValue))
        encoder.setVertexTexture(water, index: Int(waterScenePassWaterTextureIndex.rawValue))
        encoder.setFragmentBytes(&u, length: uniformLength, index: Int(waterScenePassUniformIndex.rawValue))
        encoder.drawIndexedPrimitives(type: .triangle, indexCount: planeIndexCount,
                                      indexType: .uint32, indexBuffer: planeIndexBuffer, indexBufferOffset: 0)
        encoder.endEncoding()
    }

    // MARK: - Scene encoding

    private func encodeScene(_ commandBuffer: MTLCommandBuffer) {
        guard initialized, let water = simCurrent, let caustics = causticsTexture,
              var u = currentUniforms() else { return }

        let tiles = tilesTexture ?? defaultTiles
        let sky = skyTexture ?? defaultSky
        let uniformLength = MemoryLayout<WaterSceneUniforms>.stride

        // XR: render into the engine's per-eye descriptor (drawable color+depth, set up
        // for passthrough compositing). Desktop: the water owns the frame, so build our
        // own descriptor targeting the drawable with our own depth buffer.
        let descriptor: MTLRenderPassDescriptor
        if xrMode {
            guard let xrDescriptor = renderInfo.renderPassDescriptor else { return }
            descriptor = xrDescriptor
        } else {
            guard let drawableTexture = renderInfo.renderPassDescriptor?.colorAttachments[0].texture else { return }
            ensureSceneDepth(width: drawableTexture.width, height: drawableTexture.height)
            guard let depth = sceneDepth else { return }
            let d = MTLRenderPassDescriptor()
            d.colorAttachments[0].texture = drawableTexture
            d.colorAttachments[0].loadAction = .clear
            d.colorAttachments[0].storeAction = .store
            d.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
            d.depthAttachment.texture = depth
            d.depthAttachment.loadAction = .clear
            d.depthAttachment.storeAction = .dontCare
            d.depthAttachment.clearDepth = 1.0
            descriptor = d
        }

        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }
        encoder.label = "Water Scene Pass"

        // XR: render the real-world mesh depth-only first, so real surfaces occlude the
        // water (and the sunk-in pool reads as a real hole). Meshes are world-space.
        if xrMode, let occ = occlusionRender, occ.success, let camera = CameraSystem.shared.activeCamera,
           let cc = scene.get(component: CameraComponent.self, for: camera) {
            let viewProj = simd_mul(renderInfo.perspectiveSpace, SceneRootTransform.shared.effectiveViewMatrix(cc.viewSpace))
            let meshes = occlusionLock.withLock { occlusionMeshes }
            if !meshes.isEmpty {
                encoder.setRenderPipelineState(occ.pipelineState!)
                encoder.setDepthStencilState(occ.depthState)
                encoder.setCullMode(.none)
                for m in meshes {
                    var mvp = simd_mul(viewProj, m.transform)
                    var stride = UInt32(m.vertexStride)
                    var offset = UInt32(m.vertexOffset)
                    encoder.setVertexBuffer(m.vertexBuffer, offset: 0, index: 0)
                    encoder.setVertexBytes(&stride, length: MemoryLayout<UInt32>.stride, index: 1)
                    encoder.setVertexBytes(&offset, length: MemoryLayout<UInt32>.stride, index: 2)
                    encoder.setVertexBytes(&mvp, length: MemoryLayout<simd_float4x4>.stride, index: 3)
                    encoder.drawIndexedPrimitives(type: .triangle, indexCount: m.indexCount,
                                                  indexType: m.indexType, indexBuffer: m.indexBuffer,
                                                  indexBufferOffset: m.indexOffset)
                }
            }
        }

        func bindCommonFragment() {
            encoder.setFragmentBytes(&u, length: uniformLength, index: Int(waterScenePassUniformIndex.rawValue))
            encoder.setFragmentTexture(water, index: Int(waterScenePassWaterTextureIndex.rawValue))
            encoder.setFragmentTexture(tiles, index: Int(waterScenePassTilesTextureIndex.rawValue))
            encoder.setFragmentTexture(caustics, index: Int(waterScenePassCausticsTextureIndex.rawValue))
            encoder.setFragmentTexture(sky, index: Int(waterScenePassSkyTextureIndex.rawValue))
        }

        // Pool walls + floor. The pool has inward-facing surfaces; culling the
        // camera-facing near walls (so you look INTO the pool, seeing the far walls,
        // floor, and submerged sphere) matches the original. The vertex shader's y
        // remap flips winding, so the near walls are the front-facing set here.
        if let pool = poolRender, pool.success, let poolVertexBuffer {
            encoder.setRenderPipelineState(pool.pipelineState!)
            encoder.setDepthStencilState(pool.depthState)
            encoder.setCullMode(.back)
            encoder.setVertexBuffer(poolVertexBuffer, offset: 0, index: Int(waterScenePassPositionIndex.rawValue))
            encoder.setVertexBytes(&u, length: uniformLength, index: Int(waterScenePassUniformIndex.rawValue))
            bindCommonFragment()
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: poolVertexCount)
        }

        // Sphere.
        if let sphere = sphereRender, sphere.success, let sphereVertexBuffer {
            encoder.setRenderPipelineState(sphere.pipelineState!)
            encoder.setDepthStencilState(sphere.depthState)
            encoder.setCullMode(.none)
            encoder.setVertexBuffer(sphereVertexBuffer, offset: 0, index: Int(waterScenePassPositionIndex.rawValue))
            encoder.setVertexBytes(&u, length: uniformLength, index: Int(waterScenePassUniformIndex.rawValue))
            bindCommonFragment()
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: sphereVertexCount)
        }

        // Water surface drawn twice with opposite culling so each side gets the right
        // shader: the top (above-water) face shows the refracted pool/ball + sky
        // reflection; the bottom (underwater) face is only seen if the camera dips
        // below the surface. With this plane's winding under Metal's default
        // front-facing convention, the top face is the BACK face — so the above-water
        // shader culls front, and the underwater shader culls back.
        if let below = surfaceBelowRender, below.success,
           let above = surfaceAboveRender, above.success,
           let planeVertexBuffer, let planeIndexBuffer {
            encoder.setVertexBuffer(planeVertexBuffer, offset: 0, index: Int(waterScenePassPositionIndex.rawValue))
            encoder.setVertexBytes(&u, length: uniformLength, index: Int(waterScenePassUniformIndex.rawValue))
            encoder.setVertexTexture(water, index: Int(waterScenePassWaterTextureIndex.rawValue))
            bindCommonFragment()

            encoder.setRenderPipelineState(below.pipelineState!)
            encoder.setDepthStencilState(below.depthState)
            encoder.setCullMode(.back)
            encoder.drawIndexedPrimitives(type: .triangle, indexCount: planeIndexCount,
                                          indexType: .uint32, indexBuffer: planeIndexBuffer, indexBufferOffset: 0)

            encoder.setRenderPipelineState(above.pipelineState!)
            encoder.setDepthStencilState(above.depthState)
            encoder.setCullMode(.front)
            encoder.drawIndexedPrimitives(type: .triangle, indexCount: planeIndexCount,
                                          indexType: .uint32, indexBuffer: planeIndexBuffer, indexBufferOffset: 0)
        }

        encoder.endEncoding()
    }
}

// MARK: - Public free-function API (mirrors the engine's other feature toggles)

/// Call BEFORE creating the renderer to skip the engine's deferred pipelines
/// (needed so the water-only demo fits the iOS simulator's tile-memory limit).
public func setWaterOnlyMode(_ on: Bool) { WaterRenderer.shared.waterOnlyMode = on }
/// Enable the visionOS per-eye render path. Set BEFORE creating the renderer.
public func setWaterXRMode(_ on: Bool) { WaterRenderer.shared.xrMode = on }
/// World-from-local placement of the pool (visionOS). The pool's local space is
/// [-1,1]³ with the water surface at local y=0; set this each frame from the box.
public func setWaterModelMatrix(_ m: simd_float4x4) { WaterRenderer.shared.modelMatrix = m }
public func getWaterModelMatrix() -> simd_float4x4 { WaterRenderer.shared.modelMatrix }
/// visionOS: supply the real-world meshes (scene reconstruction) that should occlude
/// the water. Call each frame (or whenever the mesh set changes).
public func setWaterOcclusionMeshes(_ meshes: [WaterOcclusionMesh]) { WaterRenderer.shared.setOcclusionMeshes(meshes) }
public func enableWater(_ enabled: Bool) { WaterRenderer.shared.setEnabled(enabled) }
public func setWaterSphere(center: simd_float3, radius: Float) { WaterRenderer.shared.setSphere(center: center, radius: radius) }
public func setWaterSphereCenter(_ center: simd_float3) { WaterRenderer.shared.setSphereCenter(center) }
public func setWaterLightDirection(_ direction: simd_float3) { WaterRenderer.shared.setLightDirection(direction) }
public func addWaterDrop(center: simd_float2, radius: Float = 0.03, strength: Float = 0.01) {
    WaterRenderer.shared.addDrop(center: center, radius: radius, strength: strength)
}
public func seedWaterRipples(count: Int = 20) { WaterRenderer.shared.seedRipples(count: count) }
public func setWaterTilesTexture(_ texture: MTLTexture) { WaterRenderer.shared.setTilesTexture(texture) }
public func setWaterSkyTexture(_ texture: MTLTexture) { WaterRenderer.shared.setSkyTexture(texture) }
public func resetWater() { WaterRenderer.shared.reset() }
