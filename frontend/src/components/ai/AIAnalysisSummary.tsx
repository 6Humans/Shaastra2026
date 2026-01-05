import { CheckCircle, XCircle, Cpu, Lightbulb } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import type { AIAnalysis } from "@/types/api"
import { cn, formatNumber, scoreToProgressColor } from "@/lib/utils"

interface AIAnalysisSummaryProps {
    analysis: AIAnalysis
}

export function AIAnalysisSummary({ analysis }: AIAnalysisSummaryProps) {
    const successRate = parseFloat(analysis.summary.success_rate.replace("%", ""))
    const avgQuality = analysis.metrics.average_quality_score * 100

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Cpu className="w-5 h-5 text-primary" />
                    AI Agent Analysis
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Processing Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-secondary/50 rounded-xl p-4 text-center">
                        <p className="text-2xl font-bold text-blue-400">
                            {formatNumber(analysis.summary.total_records_processed)}
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">Records Processed</p>
                    </div>
                    <div className="bg-secondary/50 rounded-xl p-4 text-center">
                        <div className="flex items-center justify-center gap-1">
                            <CheckCircle className="w-4 h-4 text-emerald-400" />
                            <span className="text-2xl font-bold text-emerald-400">
                                {formatNumber(analysis.summary.successful)}
                            </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Successful</p>
                    </div>
                    <div className="bg-secondary/50 rounded-xl p-4 text-center">
                        <div className="flex items-center justify-center gap-1">
                            <XCircle className="w-4 h-4 text-red-400" />
                            <span className="text-2xl font-bold text-red-400">
                                {formatNumber(analysis.summary.failed)}
                            </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Failed</p>
                    </div>
                    <div className="bg-secondary/50 rounded-xl p-4 text-center">
                        <div className="flex items-center justify-center gap-1">
                            <Lightbulb className="w-4 h-4 text-amber-400" />
                            <span className="text-2xl font-bold text-amber-400">
                                {formatNumber(analysis.metrics.total_insights_generated)}
                            </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Insights Generated</p>
                    </div>
                </div>

                {/* Success Rate & Quality Score */}
                <div className="grid md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span>Success Rate</span>
                            <span className="font-medium text-emerald-400">{analysis.summary.success_rate}</span>
                        </div>
                        <Progress
                            value={successRate}
                            max={100}
                            indicatorColor="#10b981"
                        />
                    </div>
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span>Average Quality Score</span>
                            <span
                                className="font-medium"
                                style={{ color: scoreToProgressColor(avgQuality) }}
                            >
                                {avgQuality.toFixed(1)}%
                            </span>
                        </div>
                        <Progress
                            value={avgQuality}
                            max={100}
                            indicatorColor={scoreToProgressColor(avgQuality)}
                        />
                    </div>
                </div>

                {/* Detailed Results */}
                {analysis.detailed_results.length > 0 && (
                    <div className="space-y-3">
                        <h4 className="text-sm font-medium text-muted-foreground">Analysis Details</h4>
                        <div className="grid gap-3">
                            {analysis.detailed_results.slice(0, 5).map((record) => (
                                <div
                                    key={record.record_id}
                                    className="bg-secondary/30 rounded-lg p-4 border border-border/50"
                                >
                                    <div className="flex items-center justify-between mb-3">
                                        <div className="flex items-center gap-2">
                                            <code className="text-xs bg-background px-2 py-1 rounded">
                                                {record.record_id}
                                            </code>
                                            <Badge
                                                variant={record.status === "completed" ? "success" : "destructive"}
                                            >
                                                {record.status}
                                            </Badge>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <span
                                                className={cn(
                                                    "text-lg font-bold",
                                                    record.predictions.trend_forecast === "improving" && "text-emerald-400",
                                                    record.predictions.trend_forecast === "declining" && "text-red-400",
                                                    record.predictions.trend_forecast === "stable" && "text-amber-400"
                                                )}
                                            >
                                                {record.predictions.trend_forecast === "improving" && "↗"}
                                                {record.predictions.trend_forecast === "declining" && "↘"}
                                                {record.predictions.trend_forecast === "stable" && "→"}
                                            </span>
                                            <span className="text-sm text-muted-foreground capitalize">
                                                {record.predictions.trend_forecast}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-4 gap-3 text-center text-sm">
                                        <div>
                                            <p className="text-muted-foreground text-xs">Quality</p>
                                            <p
                                                className="font-bold"
                                                style={{ color: scoreToProgressColor(record.quality_score * 100) }}
                                            >
                                                {(record.quality_score * 100).toFixed(0)}%
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-muted-foreground text-xs">7 Days</p>
                                            <p className="font-medium text-blue-400">
                                                {(record.predictions.predicted_quality_score_7d * 100).toFixed(0)}%
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-muted-foreground text-xs">14 Days</p>
                                            <p className="font-medium text-blue-400">
                                                {(record.predictions.predicted_quality_score_14d * 100).toFixed(0)}%
                                            </p>
                                        </div>
                                        <div>
                                            <p className="text-muted-foreground text-xs">30 Days</p>
                                            <p className="font-medium text-blue-400">
                                                {(record.predictions.predicted_quality_score_30d * 100).toFixed(0)}%
                                            </p>
                                        </div>
                                    </div>

                                    {record.ai_insights.length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-border/50">
                                            <p className="text-xs text-muted-foreground mb-2">Insights:</p>
                                            <ul className="space-y-1">
                                                {record.ai_insights.slice(0, 2).map((insight, i) => (
                                                    <li key={i} className="text-xs text-muted-foreground flex gap-2">
                                                        <span className="text-primary">•</span>
                                                        <span>{insight}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}

                                    {/* Detailed Predictions */}
                                    {record.predictions.predictions && record.predictions.predictions.length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-border/50">
                                            <p className="text-xs text-muted-foreground mb-2 flex items-center gap-1">
                                                <span className="text-amber-400">⚠</span> Predicted Issues:
                                            </p>
                                            <div className="space-y-2">
                                                {record.predictions.predictions.slice(0, 3).map((pred, i) => (
                                                    <div key={i} className="bg-amber-500/5 border border-amber-500/20 rounded-lg p-2">
                                                        <div className="flex items-start justify-between gap-2">
                                                            <span className="text-xs text-amber-100/90 flex-1">{pred.issue}</span>
                                                            <div className="flex items-center gap-1 flex-shrink-0">
                                                                <Badge
                                                                    variant={pred.severity === "HIGH" ? "destructive" : pred.severity === "MEDIUM" ? "warning" : "info"}
                                                                    className="text-[9px] px-1 py-0"
                                                                >
                                                                    {pred.severity}
                                                                </Badge>
                                                                <span className="text-[10px] text-muted-foreground">
                                                                    {(pred.probability * 100).toFixed(0)}%
                                                                </span>
                                                            </div>
                                                        </div>
                                                        {pred.affected_fields && pred.affected_fields.length > 0 && (
                                                            <div className="mt-1 flex flex-wrap gap-1">
                                                                {pred.affected_fields.slice(0, 3).map((field, j) => (
                                                                    <code key={j} className="text-[9px] bg-background/50 px-1 py-0.5 rounded text-purple-300">
                                                                        {field}
                                                                    </code>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
