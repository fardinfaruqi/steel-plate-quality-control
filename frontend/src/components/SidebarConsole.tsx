interface SidebarConsoleProps {
  modelInfo: Record<string, string>
  noProbabilityNote: string
  featureCount: number
}

export function SidebarConsole({
  modelInfo,
  noProbabilityNote,
  featureCount,
}: SidebarConsoleProps) {
  return (
    <div>
      <div className="console">
        <div className="mb-2 flex items-center gap-2">
          <span
            className="h-[7px] w-[7px] rounded-full bg-molten shadow-[0_0_8px_var(--color-molten)]"
            aria-hidden="true"
          />
          <span className="font-mono text-[0.7rem] font-semibold uppercase tracking-[0.18em] text-hot-white">
            Model information
          </span>
        </div>
        {Object.entries(modelInfo).map(([label, value]) => (
          <div key={label} className="console-row">
            <span className="font-mono text-[0.7rem] uppercase tracking-[0.06em] text-steel-dim">
              {label}
            </span>
            <span className="font-mono text-[0.78rem] font-semibold text-molten">
              {value}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-4 text-[0.8rem] leading-relaxed text-steel-dim">
        {noProbabilityNote}
      </p>
      <p className="mt-3 font-mono text-[0.72rem] leading-relaxed text-steel-dim">
        Model: tuned SVM on {featureCount} features (SVC, C=10, RBF kernel).
      </p>
    </div>
  )
}
