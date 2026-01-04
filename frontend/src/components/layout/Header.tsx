import { Clock, AlertTriangle, Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { useState } from "react"

interface HeaderProps {
    datasetName?: string
    timestamp?: string
    hasValidationErrors?: boolean
    onMenuToggle?: () => void
}

export function Header({
    datasetName,
    timestamp,
    hasValidationErrors = false,
    onMenuToggle
}: HeaderProps) {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

    const handleMenuToggle = () => {
        setIsMobileMenuOpen(!isMobileMenuOpen)
        onMenuToggle?.()
    }

    return (
        <header className="sticky top-0 z-30 h-16 bg-card/95 backdrop-blur border-b border-border">
            <div className="h-full flex items-center justify-between px-6">
                {/* Mobile Menu Toggle */}
                <Button
                    variant="ghost"
                    size="icon"
                    className="lg:hidden"
                    onClick={handleMenuToggle}
                >
                    {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </Button>

                {/* Dataset Info */}
                <div className="flex items-center gap-4">
                    {datasetName && (
                        <>
                            <h2 className="font-semibold text-lg hidden sm:block">{datasetName}</h2>
                            {hasValidationErrors && (
                                <Badge variant="warning" className="hidden sm:flex items-center gap-1">
                                    <AlertTriangle className="w-3 h-3" />
                                    Validation Errors
                                </Badge>
                            )}
                        </>
                    )}
                </div>

                {/* Timestamp */}
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    {timestamp && (
                        <>
                            <Clock className="w-4 h-4" />
                            <span className="hidden sm:inline">
                                {new Date(timestamp).toLocaleString()}
                            </span>
                            <span className="sm:hidden">
                                {new Date(timestamp).toLocaleTimeString()}
                            </span>
                        </>
                    )}
                </div>
            </div>
        </header>
    )
}
