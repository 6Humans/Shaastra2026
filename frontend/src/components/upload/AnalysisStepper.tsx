import { RegistrationStepper, type StepProps } from "@/components/ui/registration-stepper";
import { Loader2, CheckCircle2, LayoutDashboard, Sparkles, Scale } from "lucide-react";
import { useState, useEffect } from "react";

interface AnalysisStepperProps {
    currentStep: number;
}

const DIMENSION_LOGS = [
    { dimension: "Completeness", icon: "📊", delay: 0 },
    { dimension: "Uniqueness", icon: "🔍", delay: 300 },
    { dimension: "Validity", icon: "✅", delay: 600 },
    { dimension: "Consistency", icon: "🔗", delay: 900 },
    { dimension: "Accuracy", icon: "🎯", delay: 1200 },
    { dimension: "Timeliness", icon: "⏰", delay: 1500 },
    { dimension: "Integrity", icon: "🛡️", delay: 1800 },
];

function ProcessingLogs({ isActive }: { isActive: boolean }) {
    const [visibleLogs, setVisibleLogs] = useState<number>(0);
    const [showWeights, setShowWeights] = useState(false);

    useEffect(() => {
        if (!isActive) return;

        const timers: ReturnType<typeof setTimeout>[] = [];

        DIMENSION_LOGS.forEach((_, index) => {
            const timer = setTimeout(() => {
                setVisibleLogs(prev => Math.max(prev, index + 1));
            }, DIMENSION_LOGS[index].delay);
            timers.push(timer);
        });

        // Show weights after all dimensions
        timers.push(setTimeout(() => setShowWeights(true), 2200));

        return () => timers.forEach(clearTimeout);
    }, [isActive]);

    if (!isActive) return null;

    return (
        <div className="mt-3 space-y-2 font-mono text-xs">
            <div className="bg-black/90 dark:bg-black/80 rounded-lg p-3 border border-primary/30 max-h-48 overflow-y-auto">
                <div className="flex items-center gap-2 text-emerald-400 mb-2">
                    <Sparkles className="w-3 h-3" />
                    <span className="font-semibold">AI Quality Engine</span>
                </div>

                {DIMENSION_LOGS.slice(0, visibleLogs).map((log, idx) => (
                    <div
                        key={log.dimension}
                        className="flex items-center gap-2 text-gray-300 animate-in fade-in slide-in-from-left-2"
                        style={{ animationDelay: `${idx * 50}ms` }}
                    >
                        <span className="text-primary">▸</span>
                        <span>{log.icon}</span>
                        <span className="text-gray-400">[DIMENSION]</span>
                        <span className="text-cyan-400">Analyzing {log.dimension}...</span>
                        {idx < visibleLogs - 1 && (
                            <CheckCircle2 className="w-3 h-3 text-emerald-400 ml-auto" />
                        )}
                        {idx === visibleLogs - 1 && visibleLogs < DIMENSION_LOGS.length && (
                            <Loader2 className="w-3 h-3 animate-spin text-yellow-400 ml-auto" />
                        )}
                    </div>
                ))}

                {showWeights && (
                    <div className="mt-3 pt-2 border-t border-gray-700 animate-in fade-in">
                        <div className="flex items-center gap-2 text-amber-400 mb-1">
                            <Scale className="w-3 h-3" />
                            <span className="font-semibold">Dynamic Weight Assignment</span>
                        </div>
                        <div className="text-gray-400 text-[10px] space-y-0.5">
                            <div className="flex justify-between">
                                <span>⚖️ Applying EDA-based weights...</span>
                            </div>
                            <div className="flex justify-between text-emerald-400">
                                <span>✓ Weights normalized to 1.0</span>
                            </div>
                            <div className="flex justify-between text-cyan-400">
                                <span>⚡ Composite DQS calculated</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export function AnalysisStepper({ currentStep }: AnalysisStepperProps) {

    const steps: StepProps[] = [
        {
            step: 1,
            title: "File Upload",
            description: "Verifying file format and structure",
            content: (
                <div className="flex items-center gap-2 text-sm text-muted-foreground mt-2 bg-muted/50 p-3 rounded-lg border">
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                    <span>File received successfully. Preparing for analysis...</span>
                </div>
            ),
        },
        {
            step: 2,
            title: "Quality Check & AI Analysis",
            description: "Computing 7 Data Quality Dimensions",
            content: (
                <div className="flex flex-col gap-2 mt-2">
                    <div className="flex items-center gap-2 text-sm">
                        <Loader2 className="w-4 h-4 animate-spin text-primary" />
                        <span>Running ML-powered analysis on all dimensions...</span>
                    </div>
                    <ProcessingLogs isActive={currentStep >= 1} />
                </div>
            ),
        },
        {
            step: 3,
            title: "Finalizing Dashboard",
            description: "Structuring insights and visualizations",
            content: (
                <div className="flex items-center gap-2 text-sm text-green-600 mt-2">
                    <LayoutDashboard className="w-4 h-4" />
                    <span>Analysis complete! Redirecting to dashboard...</span>
                </div>
            ),
        },
    ];

    return (
        <div className="w-full max-w-2xl mx-auto py-10">
            <RegistrationStepper
                className="max-w-xl"
                currentStep={currentStep}
                steps={steps}
                headerTitle="Data Pipeline"
                headerStatus={currentStep >= 2 ? "Complete" : "Processing"}
            />
        </div>
    );
}
