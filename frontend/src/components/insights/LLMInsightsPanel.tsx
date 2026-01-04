import { Brain, AlertTriangle, CheckCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { LLMInsights } from "@/types/api"
import { extractPriority } from "@/lib/utils"

interface LLMInsightsPanelProps {
    insights: LLMInsights
}

export function LLMInsightsPanel({ insights }: LLMInsightsPanelProps) {
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
                <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-xl p-5 border border-purple-500/20">
                    <h4 className="text-sm font-semibold text-purple-400 mb-3">Executive Summary</h4>
                    <p className="text-sm leading-relaxed text-muted-foreground whitespace-pre-wrap">
                        {insights.summary}
                    </p>
                </div>

                {/* Root Causes */}
                {insights.root_causes.length > 0 && (
                    <div className="space-y-3">
                        <h4 className="text-sm font-semibold flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-amber-400" />
                            Root Causes Identified
                        </h4>
                        <div className="space-y-2">
                            {insights.root_causes.map((cause, index) => {
                                const priority = extractPriority(cause)
                                const cleanedCause = cause.replace(/\[(HIGH|MEDIUM|LOW)\]/gi, "").trim()

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
                                        <p className="text-sm text-muted-foreground">{cleanedCause}</p>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                )}

                {/* Recommendations */}
                {insights.recommendations.length > 0 && (
                    <div className="space-y-3">
                        <h4 className="text-sm font-semibold flex items-center gap-2">
                            <CheckCircle className="w-4 h-4 text-emerald-400" />
                            Recommended Actions
                        </h4>
                        <div className="space-y-2">
                            {insights.recommendations.map((rec, index) => {
                                const priority = extractPriority(rec)
                                const cleanedRec = rec.replace(/\[(HIGH|MEDIUM|LOW)\]/gi, "").trim()

                                return (
                                    <div
                                        key={index}
                                        className="flex items-start gap-3 bg-emerald-500/5 rounded-lg p-3 border border-emerald-500/20"
                                    >
                                        <div className="w-6 h-6 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                                            <span className="text-xs font-bold text-emerald-400">{index + 1}</span>
                                        </div>
                                        <div className="flex-1">
                                            {priority !== "MEDIUM" && (
                                                <Badge
                                                    variant={priority === "HIGH" ? "destructive" : "info"}
                                                    className="mb-2"
                                                >
                                                    {priority} Priority
                                                </Badge>
                                            )}
                                            <p className="text-sm text-muted-foreground">{cleanedRec}</p>
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
