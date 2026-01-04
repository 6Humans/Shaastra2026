import { cn, scoreToProgressColor } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface QualityScoreCardProps {
    score: number
    grade: string
    confidenceLevel: "high" | "medium" | "low"
}

export function QualityScoreCard({ score, grade, confidenceLevel }: QualityScoreCardProps) {
    const progressColor = scoreToProgressColor(score)
    const circumference = 2 * Math.PI * 45

    const confidenceColors = {
        high: "success",
        medium: "warning",
        low: "destructive",
    } as const

    return (
        <Card className="relative overflow-hidden">
            <div
                className="absolute inset-0 opacity-10"
                style={{
                    background: `radial-gradient(circle at 50% 0%, ${progressColor}, transparent 70%)`
                }}
            />
            <CardHeader className="pb-2">
                <CardTitle className="flex items-center justify-between">
                    <span>Overall Quality Score</span>
                    <Badge variant={confidenceColors[confidenceLevel]}>
                        {confidenceLevel.charAt(0).toUpperCase() + confidenceLevel.slice(1)} Confidence
                    </Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="flex items-center justify-between gap-8">
                    {/* Circular Progress */}
                    <div className="relative w-40 h-40 flex-shrink-0">
                        <svg
                            className="w-full h-full transform -rotate-90"
                            viewBox="0 0 100 100"
                        >
                            {/* Background circle */}
                            <circle
                                cx="50"
                                cy="50"
                                r="45"
                                fill="none"
                                stroke="hsl(var(--secondary))"
                                strokeWidth="8"
                            />
                            {/* Progress circle */}
                            <circle
                                cx="50"
                                cy="50"
                                r="45"
                                fill="none"
                                stroke={progressColor}
                                strokeWidth="8"
                                strokeLinecap="round"
                                strokeDasharray={circumference}
                                strokeDashoffset={circumference - (score / 100) * circumference}
                                className="transition-all duration-1000 ease-out"
                            />
                        </svg>
                        {/* Score text */}
                        <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <span
                                className="text-4xl font-bold"
                                style={{ color: progressColor }}
                            >
                                {score.toFixed(1)}
                            </span>
                            <span className="text-sm text-muted-foreground">out of 100</span>
                        </div>
                    </div>

                    {/* Grade Display */}
                    <div className="flex-1 text-center">
                        <div
                            className={cn(
                                "inline-flex items-center justify-center w-24 h-24 rounded-2xl text-5xl font-bold mb-2",
                                score >= 80 && "bg-emerald-500/20 text-emerald-400",
                                score >= 60 && score < 80 && "bg-amber-500/20 text-amber-400",
                                score < 60 && "bg-red-500/20 text-red-400"
                            )}
                        >
                            {grade.split(" ")[0]}
                        </div>
                        <p className="text-sm text-muted-foreground">
                            {grade.includes("(") ? grade.slice(grade.indexOf("(") + 1, -1) : "Quality Grade"}
                        </p>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
