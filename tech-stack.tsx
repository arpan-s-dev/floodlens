"use client"

import { motion, useReducedMotion } from "framer-motion"
import { useTheme } from "@/components/theme-provider"
import { AnimatedWave } from "@/components/home/animated-wave"

type LogoName =
  | "next"
  | "react"
  | "typescript"
  | "tailwind"
  | "framer"
  | "expo"
  | "python"
  | "pytorch"
  | "huggingface"
  | "claude"
  | "langgraph"
  | "duckdb"
  | "fastapi"
  | "postgres"
  | "supabase"
  | "docker"
  | "aws"
  | "vercel"
  | "pydantic"
  | "distilbert"
  | "sentence"
  | "parquet"
  | "tesseract"
  | "openpyxl"
  | "pdfplumber"
  | "kotlin"
  | "compose"
  | "executorch"
  | "qnn"
  | "android"
  | "tensorflow"
  | "nasa"
  | "pgvector"
  | "lora"
  | "tableau"

interface OrbitTool {
  name: string
  logo: LogoName
  color: string
}

interface OrbitGroup {
  title: string
  subtitle: string
  center: string
  rotationSeconds: number
  tools: OrbitTool[]
}

const orbitGroups: OrbitGroup[] = [
  {
    title: "AI / Agents",
    subtitle: "Model pipelines and agent tooling",
    center: "AI",
    rotationSeconds: 36,
    tools: [
      { name: "Python", logo: "python", color: "#3776AB" },
      { name: "PyTorch", logo: "pytorch", color: "#EE4C2C" },
      { name: "HuggingFace", logo: "huggingface", color: "#F59E0B" },
      { name: "Claude API", logo: "claude", color: "#D97706" },
      { name: "LangGraph", logo: "langgraph", color: "#10B981" },
      { name: "DuckDB", logo: "duckdb", color: "#F97316" },
    ],
  },
  {
    title: "Data / OCR",
    subtitle: "Document parsing and analytics exports",
    center: "OCR",
    rotationSeconds: 34,
    tools: [
      { name: "Tesseract", logo: "tesseract", color: "#A3E635" },
      { name: "pdfplumber", logo: "pdfplumber", color: "#F43F5E" },
      { name: "openpyxl", logo: "openpyxl", color: "#22C55E" },
      { name: "Parquet", logo: "parquet", color: "#06B6D4" },
      { name: "Tableau", logo: "tableau", color: "#F59E0B" },
      { name: "Pydantic", logo: "pydantic", color: "#E92063" },
    ],
  },
  {
    title: "ML / Retrieval",
    subtitle: "Matching, embeddings, and fine-tuning",
    center: "ML",
    rotationSeconds: 38,
    tools: [
      { name: "DistilBERT", logo: "distilbert", color: "#F59E0B" },
      { name: "sentence-transformers", logo: "sentence", color: "#14B8A6" },
      { name: "TensorFlow", logo: "tensorflow", color: "#FF6F00" },
      { name: "pgvector", logo: "pgvector", color: "#4169E1" },
      { name: "LoRA", logo: "lora", color: "#C084FC" },
      { name: "NASA APIs", logo: "nasa", color: "#38BDF8" },
    ],
  },
  {
    title: "Mobile / Edge",
    subtitle: "Offline Android and on-device inference",
    center: "EDGE",
    rotationSeconds: 40,
    tools: [
      { name: "Kotlin", logo: "kotlin", color: "#A855F7" },
      { name: "Jetpack Compose", logo: "compose", color: "#3DDC84" },
      { name: "ExecuTorch", logo: "executorch", color: "#EE4C2C" },
      { name: "QNN", logo: "qnn", color: "#22D3EE" },
      { name: "Android", logo: "android", color: "#3DDC84" },
      { name: "Expo", logo: "expo", color: "#8B5CF6" },
    ],
  },
  {
    title: "Frontend",
    subtitle: "Interfaces, motion, and product polish",
    center: "UI",
    rotationSeconds: 30,
    tools: [
      { name: "Next.js", logo: "next", color: "#f3edf8" },
      { name: "React", logo: "react", color: "#61DAFB" },
      { name: "TypeScript", logo: "typescript", color: "#3178C6" },
      { name: "Tailwind", logo: "tailwind", color: "#38BDF8" },
      { name: "Framer Motion", logo: "framer", color: "#EC4899" },
      { name: "Expo", logo: "expo", color: "#8B5CF6" },
    ],
  },
  {
    title: "Backend / Infra",
    subtitle: "Systems that ship and stay up",
    center: "SYS",
    rotationSeconds: 42,
    tools: [
      { name: "FastAPI", logo: "fastapi", color: "#10B981" },
      { name: "Postgres", logo: "postgres", color: "#4169E1" },
      { name: "Supabase", logo: "supabase", color: "#3ECF8E" },
      { name: "Docker", logo: "docker", color: "#2496ED" },
      { name: "AWS", logo: "aws", color: "#FF9900" },
      { name: "Vercel", logo: "vercel", color: "#A1A1AA" },
    ],
  },
]

function WordmarkLogo({
  label,
  color,
  textSize = 7,
}: {
  label: string
  color: string
  textSize?: number
}) {
  return (
    <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
      <rect x="3" y="5" width="18" height="14" rx="3" fill="none" stroke={color} strokeWidth="1.7" />
      <text
        x="12"
        y="14.5"
        textAnchor="middle"
        style={{ fill: color, fontSize: textSize, fontWeight: 800 }}
      >
        {label}
      </text>
    </svg>
  )
}

function TechLogo({ logo, color }: { logo: LogoName; color: string }) {
  switch (logo) {
    case "react":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <g fill="none" stroke={color} strokeWidth="1.5">
            <ellipse cx="12" cy="12" rx="9" ry="3.4" />
            <ellipse cx="12" cy="12" rx="9" ry="3.4" transform="rotate(60 12 12)" />
            <ellipse cx="12" cy="12" rx="9" ry="3.4" transform="rotate(120 12 12)" />
          </g>
          <circle cx="12" cy="12" r="2" fill={color} />
        </svg>
      )
    case "typescript":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2" fill={color} />
          <text x="12" y="15.5" textAnchor="middle" fill="#f3edf8" className="text-[7px] font-bold">
            TS
          </text>
        </svg>
      )
    case "tailwind":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path
            d="M4 12c2.2-4.4 5.2-5.7 9-3.8 1.4.7 2.4 1.9 4.1 1.5 1.1-.3 2-1 2.9-2.1-2.2 4.4-5.2 5.7-9 3.8-1.4-.7-2.4-1.9-4.1-1.5C5.8 10.2 4.9 10.9 4 12Zm0 5c2.2-4.4 5.2-5.7 9-3.8 1.4.7 2.4 1.9 4.1 1.5 1.1-.3 2-1 2.9-2.1-2.2 4.4-5.2 5.7-9 3.8-1.4-.7-2.4-1.9-4.1-1.5C5.8 15.2 4.9 15.9 4 17Z"
            fill={color}
          />
        </svg>
      )
    case "framer":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M6 3h12v6H6V3Zm0 6h12l-6 6H6V9Zm0 6h6v6l-6-6Z" fill={color} />
        </svg>
      )
    case "expo":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path
            d="M6 18 10.7 6.8c.5-1.2 2.1-1.2 2.6 0L18 18"
            fill="none"
            stroke={color}
            strokeWidth="2.4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )
    case "python":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M12 3c-4 0-5 1.4-5 4v2h6v1H5c-1.5 0-2.6 1.5-2.6 3.5S3.5 17 5 17h2v-3c0-2 1.5-3.5 3.5-3.5H15c1.1 0 2-.9 2-2V7c0-2.6-1-4-5-4Z" fill={color} />
          <path d="M12 21c4 0 5-1.4 5-4v-2h-6v-1h8c1.5 0 2.6-1.5 2.6-3.5S20.5 7 19 7h-2v3c0 2-1.5 3.5-3.5 3.5H9c-1.1 0-2 .9-2 2V17c0 2.6 1 4 5 4Z" fill="#F7D54A" />
          <circle cx="10" cy="6" r="1" fill="var(--bg-base)" />
          <circle cx="14" cy="18" r="1" fill="var(--bg-base)" />
        </svg>
      )
    case "pytorch":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path
            d="M13 4 8.8 8.2a6 6 0 1 0 8.5.1"
            fill="none"
            stroke={color}
            strokeWidth="2.4"
            strokeLinecap="round"
          />
          <circle cx="16.8" cy="5.8" r="1.8" fill={color} />
        </svg>
      )
    case "huggingface":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <circle cx="12" cy="12" r="8.5" fill={color} />
          <circle cx="9" cy="10" r="1.2" fill="var(--bg-base)" />
          <circle cx="15" cy="10" r="1.2" fill="var(--bg-base)" />
          <path d="M8.5 14.2c2 2 5 2 7 0" fill="none" stroke="var(--bg-base)" strokeWidth="1.7" strokeLinecap="round" />
          <path d="M4.7 8.5 2.5 6.7m16.8 1.8 2.2-1.8" stroke={color} strokeWidth="2" strokeLinecap="round" />
        </svg>
      )
    case "claude":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M12 3.5 14.1 10l6.4 2-6.4 2L12 20.5 9.9 14l-6.4-2 6.4-2L12 3.5Z" fill={color} />
          <circle cx="12" cy="12" r="2.3" fill="var(--bg-base)" opacity="0.85" />
        </svg>
      )
    case "langgraph":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <g fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round">
            <path d="M7 7h10M7 7v10M17 7v10M7 17h10M7 7l10 10" />
            <circle cx="7" cy="7" r="2.3" fill="var(--bg-base)" />
            <circle cx="17" cy="7" r="2.3" fill="var(--bg-base)" />
            <circle cx="7" cy="17" r="2.3" fill="var(--bg-base)" />
            <circle cx="17" cy="17" r="2.3" fill="var(--bg-base)" />
          </g>
        </svg>
      )
    case "duckdb":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <ellipse cx="12" cy="6" rx="6.5" ry="3" fill="none" stroke={color} strokeWidth="1.8" />
          <path d="M5.5 6v10c0 1.7 2.9 3 6.5 3s6.5-1.3 6.5-3V6" fill="none" stroke={color} strokeWidth="1.8" />
          <path d="M5.5 11c0 1.7 2.9 3 6.5 3s6.5-1.3 6.5-3" fill="none" stroke={color} strokeWidth="1.8" />
        </svg>
      )
    case "fastapi":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <circle cx="12" cy="12" r="9" fill={color} />
          <path d="M13.2 4.8 7.8 13h3.4l-1 6.2 6-8.5h-3.5l.5-5.9Z" fill="var(--bg-base)" />
        </svg>
      )
    case "postgres":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <ellipse cx="12" cy="6" rx="6.8" ry="3" fill="none" stroke={color} strokeWidth="1.8" />
          <path d="M5.2 6v10c0 1.7 3 3.1 6.8 3.1s6.8-1.4 6.8-3.1V6M5.2 11c0 1.7 3 3.1 6.8 3.1s6.8-1.4 6.8-3.1" fill="none" stroke={color} strokeWidth="1.8" />
        </svg>
      )
    case "supabase":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M13 3 5 13h6l-1 8 9-11h-6l0-7Z" fill={color} />
        </svg>
      )
    case "docker":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <g fill={color}>
            <rect x="4" y="10" width="3" height="3" rx=".4" />
            <rect x="8" y="10" width="3" height="3" rx=".4" />
            <rect x="12" y="10" width="3" height="3" rx=".4" />
            <rect x="8" y="6" width="3" height="3" rx=".4" />
            <rect x="12" y="6" width="3" height="3" rx=".4" />
            <path d="M3 14h15.5c-.7 3-3.4 5-7.6 5C7.2 19 4.5 17.5 3 14Z" />
            <path d="M18 10.5c1.3 0 2.2.7 2.9 1.8-1 .5-2.1.6-3.2.3.1-.8.2-1.5.3-2.1Z" />
          </g>
        </svg>
      )
    case "aws":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <text x="12" y="11.5" textAnchor="middle" style={{ fill: color }} className="text-[7px] font-black">
            AWS
          </text>
          <path d="M7 15c3 2 7 2 10 0" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" />
          <path d="M16.4 14.5 18 15l-.7 1.5" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      )
    case "vercel":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M12 4 22 20H2L12 4Z" fill={color} />
        </svg>
      )
    case "tableau":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <g stroke={color} strokeWidth="1.6" strokeLinecap="round">
            <path d="M12 3v5M9.5 5.5h5M12 16v5M9.5 18.5h5M3 12h5M5.5 9.5v5M16 12h5M18.5 9.5v5M6.2 6.2l3.5 3.5M8.7 4.7l-3.5 3.5M14.3 14.3l3.5 3.5M15.3 19.3l3.5-3.5" />
          </g>
        </svg>
      )
    case "android":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M7 10h10v7.2c0 1-.8 1.8-1.8 1.8H8.8C7.8 19 7 18.2 7 17.2V10Z" fill={color} />
          <path d="M8 9a4 4 0 0 1 8 0M9 5.3 7.5 3.6M15 5.3l1.5-1.7" fill="none" stroke={color} strokeWidth="1.6" strokeLinecap="round" />
          <circle cx="10" cy="13" r=".8" fill="var(--bg-base)" />
          <circle cx="14" cy="13" r=".8" fill="var(--bg-base)" />
        </svg>
      )
    case "tensorflow":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M12 3 4.5 7.2v8.6L12 20l7.5-4.2V7.2L12 3Z" fill="none" stroke={color} strokeWidth="1.7" />
          <path d="M12 3v17M4.5 7.2 12 11.5l7.5-4.3M8.2 9.4v8.4" stroke={color} strokeWidth="1.7" strokeLinecap="round" />
        </svg>
      )
    case "kotlin":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M4 4h16L12 12l8 8H4V4Z" fill={color} />
          <path d="M4 20 20 4" stroke="var(--bg-base)" strokeWidth="1.5" opacity="0.7" />
        </svg>
      )
    case "compose":
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <path d="M12 3 19.5 7.5v9L12 21l-7.5-4.5v-9L12 3Z" fill="none" stroke={color} strokeWidth="1.8" />
          <path d="M12 3v18M4.5 7.5 12 12l7.5-4.5M4.5 16.5 12 12l7.5 4.5" stroke={color} strokeWidth="1.4" />
        </svg>
      )
    case "pydantic":
      return <WordmarkLogo label="PYD" color={color} textSize={6.4} />
    case "distilbert":
      return <WordmarkLogo label="BERT" color={color} textSize={5.6} />
    case "sentence":
      return <WordmarkLogo label="ST" color={color} />
    case "parquet":
      return <WordmarkLogo label="PQ" color={color} />
    case "tesseract":
      return <WordmarkLogo label="OCR" color={color} textSize={6.5} />
    case "openpyxl":
      return <WordmarkLogo label="XL" color={color} />
    case "pdfplumber":
      return <WordmarkLogo label="PDF" color={color} textSize={6.2} />
    case "executorch":
      return <WordmarkLogo label="ET" color={color} />
    case "qnn":
      return <WordmarkLogo label="QNN" color={color} textSize={6.5} />
    case "nasa":
      return <WordmarkLogo label="NASA" color={color} textSize={5.6} />
    case "pgvector":
      return <WordmarkLogo label="VEC" color={color} textSize={6.3} />
    case "lora":
      return <WordmarkLogo label="LoRA" color={color} textSize={5.4} />
    case "next":
    default:
      return (
        <svg viewBox="0 0 24 24" className="h-6 w-6" aria-hidden="true">
          <circle cx="12" cy="12" r="9" fill="none" stroke={color} strokeWidth="1.7" opacity="0.9" />
          <path d="M8 17V7l8 10V7" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )
  }
}

function OrbitDiagram({
  group,
  reducedMotion,
  theme,
}: {
  group: OrbitGroup
  reducedMotion: boolean
  theme: "arthur" | "atie"
}) {
  const size = 240
  const center = size / 2
  const radius = 84
  const accentGlow =
    theme === "arthur"
      ? "0 0 26px rgba(201, 169, 97, 0.2)"
      : "0 0 28px rgba(168, 85, 247, 0.28)"
  const orbitPoints = group.tools.map((tool, index) => {
    const angle = (360 / group.tools.length) * index - 90
    const radians = (angle * Math.PI) / 180

    return {
      ...tool,
      angle,
      x: center + Math.cos(radians) * radius,
      y: center + Math.sin(radians) * radius,
    }
  })

  return (
    <div
      className="overflow-hidden rounded-[var(--radius-theme)] border p-4"
      style={{
        borderColor: "var(--border-faint)",
        backgroundColor: "color-mix(in srgb, var(--bg-card) 92%, transparent)",
      }}
    >
      <div className="mb-4 text-center">
        <p
          className="text-xs font-semibold uppercase tracking-[0.22em]"
          style={{ color: "var(--accent)" }}
        >
          {group.title}
        </p>
        <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
          {group.subtitle}
        </p>
      </div>

      <div className="relative mx-auto h-[240px] w-[240px]">
        <svg viewBox={`0 0 ${size} ${size}`} className="absolute inset-0 h-full w-full" aria-hidden="true">
          <circle
            cx={center}
            cy={center}
            r="96"
            fill="none"
            stroke="var(--border-faint)"
            strokeWidth="1"
            opacity="0.5"
          />
          <circle
            cx={center}
            cy={center}
            r="58"
            fill="none"
            stroke="var(--border-faint)"
            strokeWidth="1"
            opacity="0.23"
          />
          {!reducedMotion && (
            <>
              <circle cx={center} cy={center} r="34" fill="none" stroke="var(--accent)" strokeWidth="1" opacity="0">
                <animate attributeName="r" values="34;82" dur="3.4s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.34;0" dur="3.4s" repeatCount="indefinite" />
              </circle>
              <circle cx={center} cy={center} r="26" fill="none" stroke="var(--text-primary)" strokeWidth="1" opacity="0">
                <animate attributeName="r" values="26;68" dur="3.4s" begin="1.25s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.25;0" dur="3.4s" begin="1.25s" repeatCount="indefinite" />
              </circle>
            </>
          )}
          {orbitPoints.map((tool, index) => (
            <g key={`${tool.name}-line`}>
              <line
                x1={center}
                y1={center}
                x2={tool.x}
                y2={tool.y}
                stroke={tool.color}
                strokeWidth="1.2"
                strokeDasharray="4 7"
                opacity="0.36"
              >
                {!reducedMotion && (
                  <>
                    <animate
                      attributeName="stroke-dashoffset"
                      values="0;-44"
                      dur={`${2.6 + index * 0.18}s`}
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.18;0.58;0.18"
                      dur="2.4s"
                      begin={`${index * 0.18}s`}
                      repeatCount="indefinite"
                    />
                  </>
                )}
              </line>
              <circle
                cx={tool.x}
                cy={tool.y}
                r="21"
                fill="none"
                stroke={tool.color}
                strokeWidth="1"
                opacity="0.36"
              >
                {!reducedMotion && (
                  <>
                    <animate
                      attributeName="r"
                      values="19;25;19"
                      dur="2.6s"
                      begin={`${index * 0.16}s`}
                      repeatCount="indefinite"
                    />
                    <animate
                      attributeName="opacity"
                      values="0.22;0.5;0.22"
                      dur="2.6s"
                      begin={`${index * 0.16}s`}
                      repeatCount="indefinite"
                    />
                  </>
                )}
              </circle>
            </g>
          ))}
        </svg>

        <div
          className="absolute left-1/2 top-1/2 z-10 flex h-16 w-16 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border text-sm font-semibold uppercase tracking-[0.18em]"
          style={{
            borderColor: "var(--accent)",
            backgroundColor: "var(--bg-elevated)",
            color: "var(--text-primary)",
            boxShadow: accentGlow,
          }}
        >
          {group.center}
        </div>

        {orbitPoints.map((tool) => (
          <div
            key={tool.name}
            className="absolute z-20 -translate-x-1/2 -translate-y-1/2"
            style={{
              left: `${(tool.x / size) * 100}%`,
              top: `${(tool.y / size) * 100}%`,
            }}
          >
            <div
              className="flex h-11 w-11 items-center justify-center rounded-full border backdrop-blur-sm transition-transform duration-200 hover:scale-110"
              style={{
                borderColor: tool.color,
                backgroundColor: "color-mix(in srgb, var(--bg-elevated) 86%, transparent)",
                boxShadow: `0 0 0 1px ${tool.color}24, 0 0 24px ${tool.color}18`,
              }}
              title={tool.name}
              aria-label={tool.name}
            >
              <TechLogo logo={tool.logo} color={tool.color} />
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap justify-center gap-2">
        {group.tools.map((tool) => (
          <span
            key={tool.name}
            className="theme-badge border px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.14em]"
            style={{
              borderColor: "var(--border-faint)",
              backgroundColor: "var(--bg-elevated)",
              color: "var(--text-muted)",
            }}
          >
            {tool.name}
          </span>
        ))}
      </div>
    </div>
  )
}

export function TechStack() {
  const { theme } = useTheme()
  const reducedMotion = useReducedMotion()

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.5, ease: "easeOut", delay: 0.1 }}
      className="theme-card relative h-full overflow-hidden p-6 md:p-8"
    >
      <div
        className="absolute inset-x-0 top-0 h-32"
        style={{
          opacity: 0.16,
          maskImage: "linear-gradient(to bottom, black 20%, transparent 100%)",
          WebkitMaskImage: "linear-gradient(to bottom, black 20%, transparent 100%)",
        }}
      >
        <AnimatedWave />
      </div>

      <div className="relative z-10 mb-6">
        <p className="section-label mb-2">
          {theme === "arthur" ? "CHAPTER 0 — TOOLING" : "STACK MAP"}
        </p>
        <h3 className="font-display text-2xl font-semibold" style={{ color: "var(--text-primary)" }}>
          Tech Stack
        </h3>
        <p className="mt-2 max-w-2xl text-sm" style={{ color: "var(--text-muted)" }}>
          Animated logo nodes for the tools behind the portfolio, freight systems,
          and agent-heavy product work.
        </p>
      </div>

      <div className="relative z-10 grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {orbitGroups.map((group) => (
          <OrbitDiagram
            key={group.title}
            group={group}
            reducedMotion={Boolean(reducedMotion)}
            theme={theme}
          />
        ))}
      </div>
    </motion.div>
  )
}
