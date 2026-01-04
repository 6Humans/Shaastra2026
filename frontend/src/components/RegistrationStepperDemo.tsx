"use client";

import { useState, useEffect } from "react";
import { RegistrationStepper, type StepProps } from "@/components/ui/registration-stepper";
import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";

// Helper components for the demo content
const CurrencyToggle = () => (
    <div className="flex items-center text-sm border rounded-md p-1 bg-muted">
        <button className="px-3 py-1 rounded-sm bg-background text-foreground shadow-sm">ETH</button>
        <button className="px-3 py-1 text-muted-foreground">USD</button>
    </div>
);

const PriceDetail = ({ label, value }: { label: string; value: string }) => (
    <div className="flex justify-between items-center text-sm">
        <p className="text-muted-foreground">{label}</p>
        <p className="font-medium text-foreground">{value}</p>
    </div>
);

const CircularTimer = ({ timeLeft, onCancel }: { timeLeft: number; onCancel: () => void }) => {
    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const progress = ((60 - timeLeft) / 60) * circumference;

    return (
        <div className="flex flex-col items-center justify-center p-4 my-4 rounded-lg border bg-muted">
            <div className="relative h-28 w-28">
                <svg className="w-full h-full" viewBox="0 0 100 100">
                    <circle
                        className="text-border"
                        strokeWidth="8"
                        stroke="currentColor"
                        fill="transparent"
                        r={radius}
                        cx="50"
                        cy="50"
                    />
                    <motion.circle
                        className="text-green-500"
                        strokeWidth="8"
                        strokeDasharray={circumference}
                        strokeDashoffset={progress}
                        strokeLinecap="round"
                        stroke="currentColor"
                        fill="transparent"
                        r={radius}
                        cx="50"
                        cy="50"
                        style={{ rotate: -90, originX: "50%", originY: "50%" }}
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset: progress }}
                        transition={{ duration: 1, ease: "linear" }}
                    />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-3xl font-bold text-green-500">{timeLeft}</span>
                </div>
            </div>
            <p className="text-center text-sm text-muted-foreground mt-4 max-w-xs">
                This timer helps prevent others from registering the name before you do. Your name is not registered until you've completed the second transaction.
            </p>
            <Button variant="ghost" className="mt-4" onClick={onCancel}>Cancel</Button>
        </div>
    );
};


// Main Demo Component
export default function RegistrationStepperDemo() {
    const [currentStep, setCurrentStep] = useState(0);
    const [timer, setTimer] = useState(60);

    // Timer logic
    useEffect(() => {
        let interval: ReturnType<typeof setInterval>;
        if (currentStep === 1 && timer > 0) {
            interval = setInterval(() => {
                setTimer((prev) => prev - 1);
            }, 1000);
        } else if (timer === 0) {
            setCurrentStep(2); // Auto-advance to the next step
        }
        return () => clearInterval(interval);
    }, [currentStep, timer]);

    const handleCommit = () => {
        setCurrentStep(1); // Move to the timer step
    };

    const resetFlow = () => {
        setCurrentStep(0);
        setTimer(60);
    };

    const steps: StepProps[] = [
        {
            step: 1,
            title: "Commit",
            description: "Complete a transaction to begin the timer",
            content: (
                <div className="space-y-4 mt-2">
                    <div className="flex justify-between items-center p-2 border rounded-lg">
                        <p className="font-mono font-medium">20.5 Gwei</p>
                        <CurrencyToggle />
                    </div>
                    <div className="space-y-2 text-sm">
                        <PriceDetail label="1 year registration" value="0.036 ETH" />
                        <PriceDetail label="Est. network fee" value="0.0096 ETH" />
                    </div>
                    <div className="border-t pt-2">
                        <PriceDetail label="Estimated total" value="0.0457 ETH" />
                    </div>
                    <Button className="w-full" onClick={handleCommit}>
                        Commit
                    </Button>
                </div>
            ),
        },
        {
            step: 2,
            title: "Wait 60 seconds",
            description: "This timer prevents front-running",
            content: <CircularTimer timeLeft={timer} onCancel={resetFlow} />,
        },
        {
            step: 3,
            title: "Complete transaction",
            description: "Open wallet and confirm transaction",
            content: (
                <div className="mt-2 text-center">
                    <p className="text-sm text-muted-foreground mb-4">You are ready to complete the registration.</p>
                    <Button className="w-full">Secure Name</Button>
                </div>
            ),
        },
    ];

    return (
        <div className="bg-background p-10">
            <RegistrationStepper
                currentStep={currentStep}
                steps={steps}
                headerTitle="karigirwa.eth"
                headerStatus="Available"
            />
        </div>
    );
}
