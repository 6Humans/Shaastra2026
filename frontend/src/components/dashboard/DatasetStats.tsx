import { FileText, Columns, AlertTriangle, Copy } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn, formatNumber, formatPercentage } from "@/lib/utils"

interface DatasetStatsProps {
    rows: number
    columns: number
    missingColumnsCount: number
    duplicatesCount: number
    duplicatesPercentage: number
}

export function DatasetStats({
    rows,
    columns,
    missingColumnsCount,
    duplicatesCount,
    duplicatesPercentage,
}: DatasetStatsProps) {
    const stats = [
        {
            label: "Total Records",
            value: formatNumber(rows),
            icon: FileText,
            color: "text-blue-400",
            bgColor: "bg-blue-500/20",
        },
        {
            label: "Columns",
            value: formatNumber(columns),
            icon: Columns,
            color: "text-purple-400",
            bgColor: "bg-purple-500/20",
        },
        {
            label: "Columns with Missing",
            value: formatNumber(missingColumnsCount),
            icon: AlertTriangle,
            color: missingColumnsCount > 0 ? "text-amber-400" : "text-emerald-400",
            bgColor: missingColumnsCount > 0 ? "bg-amber-500/20" : "bg-emerald-500/20",
        },
        {
            label: "Duplicates",
            value: `${formatNumber(duplicatesCount)} (${formatPercentage(duplicatesPercentage)})`,
            icon: Copy,
            color: duplicatesCount > 0 ? "text-amber-400" : "text-emerald-400",
            bgColor: duplicatesCount > 0 ? "bg-amber-500/20" : "bg-emerald-500/20",
        },
    ]

    return (
        <Card>
            <CardHeader>
                <CardTitle>Dataset Statistics</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    {stats.map((stat) => (
                        <div
                            key={stat.label}
                            className="bg-secondary/50 rounded-xl p-4 flex flex-col gap-3"
                        >
                            <div className="flex items-center gap-3">
                                <div className={cn("p-2 rounded-lg", stat.bgColor)}>
                                    <stat.icon className={cn("w-5 h-5", stat.color)} />
                                </div>
                                <span className="text-sm text-muted-foreground">{stat.label}</span>
                            </div>
                            <span className={cn("text-2xl font-bold", stat.color)}>
                                {stat.value}
                            </span>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    )
}
