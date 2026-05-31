import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Grid, OrbitControls } from '@react-three/drei'
import { useAppContext } from '../App'

const ROOM_CX = 2.5
const ROOM_CZ = 2.0
function toThree(p) {
  return [p.x - ROOM_CX, p.z, p.y - ROOM_CZ]
}

// Solid capsule shape: upright when standing, rotates flat when fallen/lying.
function PersonShape({ color, lying, motion }) {
  const group = useRef()
  useFrame(() => {
    if (!group.current) return
    const targetRot = lying * (Math.PI / 2)
    group.current.rotation.z += (targetRot - group.current.rotation.z) * 0.15
    const pulse = 1 + motion * 0.08 * Math.sin(performance.now() / 100)
    group.current.scale.setScalar(pulse)
  })
  return (
    <group ref={group}>
      {/* body — tall rounded cylinder */}
      <mesh position={[0, 0.55, 0]}>
        <capsuleGeometry args={[0.18, 0.75, 8, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} roughness={0.4} />
      </mesh>
    </group>
  )
}

function Blob() {
  const { frame } = useAppContext()
  const holder = useRef()
  const glow = useRef()

  useFrame(() => {
    if (!holder.current) return
    const src = frame?.position
    if (!src) return
    const [x, y, z] = toThree(src)
    const m = frame?.motion ?? 0
    const jitter = 0.018 + m * 0.06
    const nx = x + (Math.random() - 0.5) * jitter
    const nz = z + (Math.random() - 0.5) * jitter
    holder.current.position.x += (nx - holder.current.position.x) * 0.38
    holder.current.position.z += (nz - holder.current.position.z) * 0.38
    holder.current.position.y += (y - holder.current.position.y) * 0.4

    if (glow.current) {
      glow.current.material.opacity = 0.1 + (frame?.motion ?? 0) * 0.2
    }
  })

  if (!frame?.position) return null

  const posture = frame?.posture ?? 'standing'
  const lying = posture === 'lying' || posture === 'fallen' ? 1 : posture === 'lying-down' ? 0.55 : 0
  const color = posture === 'fallen' ? '#ef4444' : '#22d3ee'

  return (
    <group ref={holder}>
      <mesh ref={glow} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <circleGeometry args={[0.55, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.15} />
      </mesh>
      <PersonShape color={color} lying={lying} motion={frame?.motion ?? 0} />
    </group>
  )
}

export default function Scene3D() {
  const { frame, connected } = useAppContext()
  const hasData = !!frame?.position

  return (
    <div className="relative h-[420px] rounded-2xl overflow-hidden bg-stone-900 border border-stone-700">
      <Canvas camera={{ position: [4, 4, 6], fov: 50 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 8, 5]} intensity={0.8} />
        <Grid
          args={[14, 14]} cellSize={0.5} cellColor="#3f3f46"
          sectionSize={2} sectionColor="#52525b" infiniteGrid fadeDistance={26}
        />
        <Blob />
        <OrbitControls enablePan={false} minDistance={3} maxDistance={14} maxPolarAngle={Math.PI / 2.1} />
      </Canvas>

      <div className="absolute top-2 left-3 text-xs text-stone-400 font-mono">
        WiFi position estimate · floor view
      </div>

      {!hasData && !connected && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 pointer-events-none">
          <div className="w-2 h-2 rounded-full bg-stone-500 animate-pulse" />
          <p className="text-stone-500 text-sm">No signal</p>
        </div>
      )}
    </div>
  )
}
