import type { FeatureGroup, FeatureKind } from '../types'
import { FeatureInput } from './FeatureInput'

interface FeaturePanelProps {
  group: FeatureGroup
  kinds: Map<string, FeatureKind>
  values: Record<string, number>
  onValueChange: (feature: string, value: number) => void
}

export function FeaturePanel({
  group,
  kinds,
  values,
  onValueChange,
}: FeaturePanelProps) {
  return (
    <section
      className="panel mb-4 p-4"
      aria-label={`${group.name} measurements`}
    >
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h4 className="font-display text-[0.92rem] font-extrabold uppercase tracking-[0.12em] text-hot-white [font-stretch:125%]">
          {group.name}
        </h4>
        <span className="whitespace-nowrap font-mono text-[0.68rem] uppercase tracking-[0.14em] text-molten">
          {String(group.features.length).padStart(2, '0')} inputs
        </span>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {group.features.map((feature) => (
          <FeatureInput
            key={feature}
            feature={feature}
            kind={kinds.get(feature) ?? 'decimal'}
            value={values[feature] ?? 0}
            onChange={(value) => onValueChange(feature, value)}
          />
        ))}
      </div>
    </section>
  )
}
