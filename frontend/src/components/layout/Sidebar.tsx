import {
    LayoutDashboard,
    AlertTriangle,
    Cpu,
    Brain,
    Upload,
    BarChart3,
    Github,
    Database
} from "lucide-react"
import { cn } from "@/lib/utils"

interface SidebarProps {
    activeSection: string
    onSectionChange: (section: string) => void
    hasData: boolean
}

const navItems = [
    { id: "upload", label: "Upload Data", icon: Upload, requiresData: false },
    { id: "overview", label: "Overview", icon: LayoutDashboard, requiresData: true },
    { id: "dimensions", label: "Data Quality", icon: BarChart3, requiresData: true },
    { id: "anomalies", label: "Anomalies", icon: AlertTriangle, requiresData: true },
    { id: "ai-analysis", label: "AI Analysis", icon: Cpu, requiresData: true },
    { id: "insights", label: "Insights", icon: Brain, requiresData: true },
]

export function Sidebar({ activeSection, onSectionChange, hasData }: SidebarProps) {
    return (
        <aside className="fixed left-0 top-0 z-40 h-screen w-64 bg-card border-r border-border flex flex-col">
            {/* Logo */}
            <div className="h-16 flex items-center gap-3 px-6 border-b border-border">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <Database className="w-5 h-5 text-white" />
                </div>
                <div>
                    <h1 className="font-bold text-lg gradient-text">DataQuality</h1>
                    <p className="text-xs text-muted-foreground">AI Analysis</p>
                </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-4 py-6 space-y-2">
                {navItems.map((item) => {
                    const isDisabled = item.requiresData && !hasData
                    const isActive = activeSection === item.id

                    return (
                        <button
                            key={item.id}
                            onClick={() => !isDisabled && onSectionChange(item.id)}
                            disabled={isDisabled}
                            className={cn(
                                "w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200",
                                isActive
                                    ? "bg-primary text-primary-foreground shadow-lg shadow-primary/25"
                                    : "text-muted-foreground hover:text-foreground hover:bg-secondary",
                                isDisabled && "opacity-40 cursor-not-allowed hover:bg-transparent"
                            )}
                        >
                            <item.icon className="w-5 h-5" />
                            {item.label}
                        </button>
                    )
                })}
            </nav>

            {/* Footer */}
            <div className="p-4 border-t border-border">
                <a
                    href="https://github.com"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                    <Github className="w-4 h-4" />
                    View on GitHub
                </a>
            </div>
        </aside>
    )
}
