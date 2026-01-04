import { AlertTriangle, AlertCircle, AlertOctagon, Info, TrendingDown } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import type { AnomalySummary as AnomalySummaryType } from "@/types/api"
import { cn, formatNumber, formatPercentage } from "@/lib/utils"

interface AnomalySummaryProps {
    summary: AnomalySummaryType
}

export function AnomalySummary({ summary }: AnomalySummaryProps) {
    const stats = [
        {
            label: "Total Anomalies",
            value: formatNumber(summary.total_anomalies),
            icon: AlertTriangle,
            color: "text-amber-400",
            bgColor: "bg-amber-500/20",
        },
        {
            label: "Critical Issues",
            value: formatNumber(summary.critical_count),
            icon: AlertOctagon,
            color: summary.critical_count > 0 ? "text-red-400" : "text-emerald-400",
            bgColor: summary.critical_count > 0 ? "bg-red-500/20" : "bg-emerald-500/20",
        },
        {
            label: "Errors",
            value: formatNumber(summary.error_count),
            icon: AlertCircle,
            color: summary.error_count > 0 ? "text-orange-400" : "text-emerald-400",
            bgColor: summary.error_count > 0 ? "bg-orange-500/20" : "bg-emerald-500/20",
        },
        {
            label: "Warnings",
            value: formatNumber(summary.warning_count),
            icon: Info,
            color: summary.warning_count > 0 ? "text-amber-400" : "text-emerald-400",
            bgColor: summary.warning_count > 0 ? "bg-amber-500/20" : "bg-emerald-500/20",
        },
        {
            label: "Outliers",
            value: formatNumber(summary.outlier_count),
            icon: TrendingDown,
            color: summary.outlier_count > 0 ? "text-purple-400" : "text-emerald-400",
            bgColor: summary.outlier_count > 0 ? "bg-purple-500/20" : "bg-emerald-500/20",
        },
    ]

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span>Anomaly Summary</span>
                    <span className="text-sm font-normal text-muted-foreground">
                        {formatPercentage(summary.overall_anomaly_rate)} of data
                    </span>
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                {summary.has_critical_issues && (
                    <Alert variant="destructive">
                        <AlertTitle>🚨 Critical Issues Detected</AlertTitle>
                        <AlertDescription>
                            Processing may be affected. Review critical failures for details.
                        </AlertDescription>
                    </Alert>
                )}

                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                    {stats.map((stat) => (
                        <div
                            key={stat.label}
                            className="bg-secondary/50 rounded-xl p-4 text-center"
                        >
                            <div className={cn("inline-flex p-2 rounded-lg mb-2", stat.bgColor)}>
                                <stat.icon className={cn("w-5 h-5", stat.color)} />
                            </div>
                            <p className={cn("text-2xl font-bold", stat.color)}>{stat.value}</p>
                            <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    )
}
