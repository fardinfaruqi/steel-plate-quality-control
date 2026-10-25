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
    <section aria-label={`${group.name} measurements`}>
      <div className="group__head">
        <h3 className="group__title">{group.name}</h3>
        <span className="group__count">
          {group.features.length} {group.features.length === 1 ? 'input' : 'inputs'}
        </span>
      </div>
      <div className="group__fields">
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