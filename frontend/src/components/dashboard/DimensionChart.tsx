import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    ResponsiveContainer,
    Cell,
    Tooltip as RechartsTooltip,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
    Radar
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { DimensionScores } from "@/types/api"
import { scoreToProgressColor } from "@/lib/utils"

interface DimensionChartProps {
    scores: DimensionScores
    chartType?: "bar" | "radar"
}

const dimensionLabels: Record<keyof DimensionScores, string> = {
    completeness: "Completeness",
    uniqueness: "Uniqueness",
    validity: "Validity",
    consistency: "Consistency",
    accuracy: "Accuracy",
    timeliness: "Timeliness",
}

const dimensionDescriptions: Record<keyof DimensionScores, string> = {
    completeness: "% of non-null values",
    uniqueness: "% of non-duplicate records",
    validity: "% passing format checks",
    consistency: "% following patterns",
    accuracy: "Outlier detection score",
    timeliness: "Data freshness score",
}

export function DimensionChart({ scores, chartType = "bar" }: DimensionChartProps) {
    const data = Object.entries(scores).map(([key, value]) => ({
        name: dimensionLabels[key as keyof DimensionScores],
        fullName: key,
        value: typeof value === 'number' ? value : 0,
        description: dimensionDescriptions[key as keyof DimensionScores],
        color: scoreToProgressColor(typeof value === 'number' ? value : 0),
    }))

    const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ payload: typeof data[0] }> }) => {
        if (active && payload && payload.length) {
            const item = payload[0].payload
            return (
                <div className="bg-card border border-border rounded-lg px-4 py-3 shadow-xl">
                    <p className="font-semibold">{item.name}</p>
                    <p className="text-2xl font-bold" style={{ color: item.color }}>
                        {item.value.toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground">{item.description}</p>
                </div>
            )
        }
        return null
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle>Data Quality Dimensions</CardTitle>
            </CardHeader>
            <CardContent>
                {chartType === "bar" ? (
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={data} layout="vertical">
                            <CartesianGrid
                                strokeDasharray="3 3"
                                horizontal={true}
                                vertical={false}
                                stroke="hsl(var(--border))"
                            />
                            <XAxis
                                type="number"
                                domain={[0, 100]}
                                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                                axisLine={{ stroke: "hsl(var(--border))" }}
                            />
                            <YAxis
                                type="category"
                                dataKey="name"
                                width={100}
                                tick={{ fill: "hsl(var(--foreground))", fontSize: 12 }}
                                axisLine={{ stroke: "hsl(var(--border))" }}
                            />
                            <RechartsTooltip content={<CustomTooltip />} />
                            <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={30}>
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                ) : (
                    <ResponsiveContainer width="100%" height={300}>
                        <RadarChart data={data}>
                            <PolarGrid stroke="hsl(var(--border))" />
                            <PolarAngleAxis
                                dataKey="name"
                                tick={{ fill: "hsl(var(--foreground))", fontSize: 11 }}
                            />
                            <PolarRadiusAxis
                                domain={[0, 100]}
                                tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                            />
                            <Radar
                                name="Score"
                                dataKey="value"
                                stroke="hsl(var(--primary))"
                                fill="hsl(var(--primary))"
                                fillOpacity={0.3}
                                strokeWidth={2}
                            />
                        </RadarChart>
                    </ResponsiveContainer>
                )}

                {/* Legend with score pills */}
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">
                    {data.map((item) => (
                        <div
                            key={item.fullName}
                            className="flex items-center justify-between bg-secondary/50 rounded-lg px-3 py-2"
                        >
                            <span className="text-sm truncate">{item.name}</span>
                            <span
                                className="font-bold text-sm ml-2"
                                style={{ color: item.color }}
                            >
                                {item.value.toFixed(1)}
                            </span>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    )
}
