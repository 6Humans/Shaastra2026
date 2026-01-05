import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import type { ColumnOutlier } from "@/types/api"
import { formatNumber, formatPercentage } from "@/lib/utils"

interface OutlierTableProps {
    outliers: ColumnOutlier[]
    totalOutliers: number
}

export function OutlierTable({ outliers, totalOutliers }: OutlierTableProps) {
    if (!outliers || outliers.length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>ML-Detected Outliers</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="py-8 text-center text-muted-foreground">
                        <p>✨ No significant outliers detected</p>
                        <p className="text-sm mt-2">Data appears to be within expected ranges.</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span>ML-Detected Outliers</span>
                    <Badge variant="warning">{formatNumber(totalOutliers)} total</Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Column</TableHead>
                            <TableHead>Count</TableHead>
                            <TableHead>Percentage</TableHead>
                            <TableHead>Severity</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {outliers.map((outlier) => (
                            <TableRow key={outlier.column}>
                                <TableCell className="font-medium">
                                    <code className="text-sm bg-secondary px-2 py-1 rounded">
                                        {outlier.column}
                                    </code>
                                </TableCell>
                                <TableCell>{formatNumber(outlier.outlier_count)}</TableCell>
                                <TableCell>
                                    <span className={outlier.outlier_percentage > 10 ? "text-red-400" : "text-amber-400"}>
                                        {formatPercentage(outlier.outlier_percentage)}
                                    </span>
                                </TableCell>
                                <TableCell>
                                    <Badge
                                        variant={outlier.severity === "high" ? "destructive" : "warning"}
                                    >
                                        {outlier.severity}
                                    </Badge>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </CardContent>
        </Card>
    )
}

