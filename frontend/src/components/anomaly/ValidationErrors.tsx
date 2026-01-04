import { AlertCircle } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { ValidationError } from "@/types/api"
import { cn, severityToColor } from "@/lib/utils"

interface ValidationErrorsProps {
    errors: ValidationError[]
    totalErrors: number
}

export function ValidationErrors({ errors, totalErrors }: ValidationErrorsProps) {
    if (errors.length === 0) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Type Validation Errors</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="py-8 text-center text-muted-foreground">
                        <p>✅ No validation errors found</p>
                        <p className="text-sm mt-2">All data types are correctly formatted.</p>
                    </div>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span>Type Validation Errors</span>
                    <Badge variant="destructive">{totalErrors} errors</Badge>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    {errors.map((error, index) => (
                        <div
                            key={index}
                            className={cn(
                                "border rounded-lg p-4",
                                severityToColor(error.severity)
                            )}
                        >
                            <div className="flex items-start gap-3">
                                <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2 flex-wrap mb-2">
                                        <code className="text-sm bg-background/50 px-2 py-1 rounded font-semibold">
                                            {error.column}
                                        </code>
                                        <Badge variant="outline" className="text-xs">
                                            {error.error_type}
                                        </Badge>
                                        <Badge variant="outline" className="text-xs">
                                            {error.dimension}
                                        </Badge>
                                    </div>
                                    <p className="text-sm mb-2">{error.message}</p>
                                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                                        <span>Count: {error.count}</span>
                                        <span>({error.percentage.toFixed(2)}%)</span>
                                    </div>
                                    {error.sample_invalid_values.length > 0 && (
                                        <div className="mt-2 pt-2 border-t border-current/20">
                                            <span className="text-xs font-medium">Sample values: </span>
                                            <code className="text-xs bg-background/50 px-1.5 py-0.5 rounded">
                                                {error.sample_invalid_values
                                                    .slice(0, 5)
                                                    .map(v => String(v))
                                                    .join(", ")}
                                            </code>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
    )
}
