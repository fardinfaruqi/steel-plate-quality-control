export interface FeatureGroup {
  name: string
  slug: string
  features: string[]
}

export interface Metadata {
  feature_order: string[]
  groups: FeatureGroup[]
  integer_features: string[]
  binary_features: string[]
  model_info: Record<string, string>
  about_model: Record<string, string>
  disclaimer: string
  no_probability_note: string
  fault_types: string[]
}

export interface ExampleRow {
  index: number
  defect: string
  values: number[]
}

export interface Examples {
  total: number
  feature_order: string[]
  rows: ExampleRow[]
}

export type FeatureKind = 'binary' | 'integer' | 'decimal'
