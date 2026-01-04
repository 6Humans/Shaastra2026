import { RegistrationStepper, type StepProps } from "@/components/ui/registration-stepper";
import { Loader2, CheckCircle2, LayoutDashboard } from "lucide-react";

interface AnalysisStepperProps {
    currentStep: number;
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
            description: "Detecting outliers, missing values, and patterns",
            content: (
                <div className="flex flex-col gap-3 mt-2">
                    <div className="flex items-center gap-2 text-sm">
                        <Loader2 className="w-4 h-4 animate-spin text-primary" />
                        <span>Running deep learning models...</span>
                    </div>
                    <div className="space-y-1">
                        <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                            <div className="h-full bg-primary animate-progress origin-left w-full" style={{ animationDuration: '2s' }}></div>
                        </div>
                        <p className="text-xs text-muted-foreground text-right">Analyzing data points...</p>
                    </div>
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
