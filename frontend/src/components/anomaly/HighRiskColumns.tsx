import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import type { HighRiskColumn } from "@/types/api"
import { severityToColor, cn } from "@/lib/utils"

interface HighRiskColumnsProps {
    columns: HighRiskColumn[]
    maxItems?: number
}

export function HighRiskColumns({ columns, maxItems = 10 }: HighRiskColumnsProps) {
    const displayedColumns = columns.slice(0, maxItems)

    if (columns.length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>High-Risk Columns</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="py-8 text-center text-muted-foreground">
                        <p>🎉 No high-risk columns detected!</p>
                        <p className="text-sm mt-2">Your data quality looks great.</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span>High-Risk Columns</span>
                    <Badge variant="outline">{columns.length} columns</Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Column</TableHead>
                            <TableHead>Risk Score</TableHead>
                            <TableHead className="hidden md:table-cell">Issues</TableHead>
                            <TableHead>Severity</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {displayedColumns.map((col) => (
                            <TableRow key={col.column}>
                                <TableCell className="font-medium">
                                    <code className="text-sm bg-secondary px-2 py-1 rounded">
                                        {col.column}
                                    </code>
                                </TableCell>
                                <TableCell>
                                    <div className="flex items-center gap-3 min-w-32">
                                        <Progress
                                            value={col.risk_score}
                                            max={100}
                                            indicatorColor={
                                                col.risk_score >= 70 ? "#ef4444" :
                                                    col.risk_score >= 40 ? "#f59e0b" : "#10b981"
                                            }
                                            className="flex-1"
                                        />
                                        <span className="text-sm font-medium w-12">
                                            {col.risk_score.toFixed(0)}
                                        </span>
                                    </div>
                                </TableCell>
                                <TableCell className="hidden md:table-cell">
                                    <div className="flex flex-wrap gap-1">
                                        {col.anomaly_types.slice(0, 3).map((type, i) => (
                                            <Badge key={i} variant="secondary" className="text-xs">
                                                {type.length > 20 ? type.slice(0, 20) + "..." : type}
                                            </Badge>
                                        ))}
                                        {col.anomaly_types.length > 3 && (
                                            <Badge variant="outline" className="text-xs">
                                                +{col.anomaly_types.length - 3}
                                            </Badge>
                                        )}
                                    </div>
                                </TableCell>
                                <TableCell>
                                    <Badge
                                        className={cn("border", severityToColor(col.severity))}
                                    >
                                        {col.severity}
                                    </Badge>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
                {columns.length > maxItems && (
                    <p className="text-sm text-muted-foreground text-center mt-4">
                        Showing {maxItems} of {columns.length} columns
                    </p>
                )}
            </CardContent>
        </Card>
    )
}
