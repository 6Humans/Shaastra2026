import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

export function scoreToColor(score: number): string {
    if (score >= 80) return "text-emerald-400"
    if (score >= 60) return "text-amber-400"
    return "text-red-400"
}

export function scoreToBgColor(score: number): string {
    if (score >= 80) return "bg-emerald-500/20 text-emerald-400"
    if (score >= 60) return "bg-amber-500/20 text-amber-400"
    return "bg-red-500/20 text-red-400"
}

export function scoreToProgressColor(score: number): string {
    if (score >= 80) return "#10b981"
    if (score >= 60) return "#f59e0b"
    return "#ef4444"
}

export function formatNumber(num: number): string {
    return new Intl.NumberFormat().format(num)
}

export function formatPercentage(num: number): string {
    return `${num.toFixed(2)}%`
}

export function extractPriority(text: string): string {
    const match = text.match(/\[(HIGH|MEDIUM|LOW)\]/i)
    return match ? match[1].toUpperCase() : "MEDIUM"
}

export function severityToColor(severity: string): string {
    switch (severity.toLowerCase()) {
        case "critical":
            return "bg-red-500/20 text-red-400 border-red-500/50"
        case "high":
            return "bg-orange-500/20 text-orange-400 border-orange-500/50"
        case "medium":
            return "bg-amber-500/20 text-amber-400 border-amber-500/50"
        case "low":
            return "bg-blue-500/20 text-blue-400 border-blue-500/50"
        default:
            return "bg-slate-500/20 text-slate-400 border-slate-500/50"
    }
}

export function trendToIcon(trend: string): { icon: string; color: string } {
    switch (trend.toLowerCase()) {
        case "improving":
            return { icon: "↗", color: "text-emerald-400" }
        case "declining":
            return { icon: "↘", color: "text-red-400" }
        default:
            return { icon: "→", color: "text-amber-400" }
    }
}
