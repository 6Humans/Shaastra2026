import { Brain, AlertTriangle, CheckCircle, Info } from "lucide-react"
import ReactMarkdown from "react-markdown"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { LLMInsights, RootCause } from "@/types/api"

interface LLMInsightsPanelProps {
    insights: LLMInsights | null | undefined
}

export function LLMInsightsPanel({ insights }: LLMInsightsPanelProps) {
    // Handle null/undefined insights
    if (!insights) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Brain className="w-5 h-5 text-purple-400" />
                        AI-Generated Insights
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                        <Info className="w-12 h-12 text-muted-foreground/50 mb-4" />
                        <p className="text-muted-foreground">No insights available yet.</p>
                        <p className="text-sm text-muted-foreground/70 mt-1">Upload a file to generate AI insights.</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    // Helper to extract priority from root cause
    const getRootCauseDetails = (cause: string | RootCause) => {
        if (typeof cause === 'string') {
            const priorityMatch = cause.match(/\[(HIGH|MEDIUM|LOW)\]/i)
            const priority = priorityMatch ? priorityMatch[1].toUpperCase() : 'MEDIUM'
            const cleanText = cause
                .replace(/\[(HIGH|MEDIUM|LOW)\]/gi, "")
                .replace(/priority/gi, "")
                .replace(/^\*\*|\*\*$/g, "")
                .replace(/^-\s*/, "")
                .trim()
            return { priority, text: cleanText, field: null }
        } else {
            return {
                priority: cause.severity || 'MEDIUM',
                text: cause.cause || '',
                field: cause.field || null
            }
        }
    }

    const hasRootCauses = insights.root_causes && insights.root_causes.length > 0
    const hasRecommendations = insights.recommended_actions && insights.recommended_actions.length > 0

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Brain className="w-5 h-5 text-purple-400" />
                    AI-Generated Insights
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Executive Summary */}
                {insights.summary && (
                    <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-xl p-5 border border-purple-500/20">
                        <h4 className="text-sm font-semibold text-purple-400 mb-3">Executive Summary</h4>
                        <div className="text-sm leading-relaxed text-muted-foreground [&>p]:mb-2 [&>p:last-child]:mb-0">
                            <ReactMarkdown>{insights.summary}</ReactMarkdown>
                        </div>
                    </div>
                )}

                {/* Root Causes */}
                {hasRootCauses && (
                    <div className="space-y-3">
                        <h4 className="text-sm font-semibold flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-amber-400" />
                            Root Causes Identified
                        </h4>
                        <div className="space-y-2">
                            {insights.root_causes.map((cause, index) => {
                                const { priority, text, field } = getRootCauseDetails(cause)
                                return (
                                    <div
                                        key={index}
                                        className="flex items-start gap-3 bg-secondary/50 rounded-lg p-3"
                                    >
                                        <Badge
                                            variant={
                                                priority === "HIGH" ? "destructive" :
                                                    priority === "MEDIUM" ? "warning" : "info"
                                            }
                                            className="mt-0.5 flex-shrink-0"
                                        >
                                            {priority}
                                        </Badge>
                                        <div className="flex-1">
                                            {field && (
                                                <code className="text-xs text-purple-300 bg-purple-500/10 px-1.5 py-0.5 rounded mb-1 inline-block">
                                                    {field}
                                                </code>
                                            )}
                                            <div className="text-sm text-muted-foreground [&>p]:mb-1 [&>p:last-child]:mb-0">
                                                <ReactMarkdown>{text}</ReactMarkdown>
                                            </div>
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                )}

                {/* Recommended Actions */}
                {hasRecommendations && (
                    <div className="space-y-3">
                        <h4 className="text-sm font-semibold flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-emerald-400" />
                            Recommended Actions
                        </h4>
                        <div className="space-y-3">
                            {insights.recommended_actions.map((action, index) => (
                                <div
                                    key={index}
                                    className="flex flex-col gap-2 bg-emerald-500/5 rounded-lg p-3 border border-emerald-500/20"
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <div className="w-5 h-5 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
                                                <span className="text-xs font-bold text-emerald-400">{index + 1}</span>
                                            </div>
                                            <span className="text-sm font-medium text-emerald-100">
                                                {action.action}
                                            </span>
                                        </div>
                                        <Badge
                                            variant={action.priority === "HIGH" ? "destructive" : action.priority === "MEDIUM" ? "warning" : "info"}
                                            className="uppercase text-[10px]"
                                        >
                                            {action.priority}
                                        </Badge>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 pl-7 text-xs">
                                        <div className="bg-background/50 p-2 rounded border border-border/50">
                                            <span className="text-muted-foreground block mb-0.5">Field</span>
                                            <code className="text-purple-300 bg-purple-500/10 px-1 py-0.5 rounded">{action.field}</code>
                                        </div>
                                        <div className="bg-background/50 p-2 rounded border border-border/50">
                                            <span className="text-muted-foreground block mb-0.5">Why</span>
                                            <span className="text-amber-200/80">{action.why}</span>
                                        </div>
                                        <div className="bg-background/50 p-2 rounded border border-border/50">
                                            <span className="text-muted-foreground block mb-0.5">Impact</span>
                                            <span className="text-red-200/80">{action.impact}</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* No data message */}
                {!insights.summary && !hasRootCauses && !hasRecommendations && (
                    <div className="flex flex-col items-center justify-center py-8 text-center">
                        <Info className="w-10 h-10 text-muted-foreground/50 mb-3" />
                        <p className="text-muted-foreground">LLM analysis did not return insights.</p>
                        <p className="text-sm text-muted-foreground/70 mt-1">Try re-uploading the file.</p>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
