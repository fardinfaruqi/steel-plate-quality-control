import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import type { FeatureKind } from '../types'

interface FeatureInputProps {
  feature: string
  kind: FeatureKind
  value: number
  onChange: (value: number) => void
}

function format(value: number, precision: number | undefined): string {
  return precision === undefined ? String(value) : value.toFixed(precision)
}

function NumberField({
  value,
  step,
  precision,
  onChange,
}: {
  value: number
  step: number
  precision: number | undefined
  onChange: (value: number) => void
}) {
  const [raw, setRaw] = useState(() => format(value, precision))
  const ref = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (document.activeElement !== ref.current) {
      setRaw(format(value, precision))
    }
  }, [value, precision])

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    setRaw(e.target.value)
    const parsed = Number(e.target.value)
    if (e.target.value !== '' && !Number.isNaN(parsed)) {
      onChange(parsed)
    }
  }

  function handleBlur() {
    const parsed = Number(raw)
    if (raw !== '' && !Number.isNaN(parsed)) {
      setRaw(format(parsed, precision))
    } else {
      setRaw(format(value, precision))
    }
  }

  return (
    <input
      ref={ref}
      type="number"
      className="field__input"
      value={raw}
      step={step}
      onChange={handleChange}
      onBlur={handleBlur}
      inputMode="decimal"
    />
  )
}

export function FeatureInput({
  feature,
  kind,
  value,
  onChange,
}: FeatureInputProps) {
  const id = `feature-${feature}`

  if (kind === 'binary') {
    return (
      <label htmlFor={id} className="field">
        <span className="field__label">{feature}</span>
        <span className="field__wrap field__wrap--select">
          <select
            id={id}
            className="field__input field__input--select"
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
          >
            <option value={0}>0</option>
            <option value={1}>1</option>
          </select>
        </span>
      </label>
    )
  }

  const integer = kind === 'integer'

  return (
    <label htmlFor={id} className="field">
      <span className="field__label">{feature}</span>
      <span className="field__wrap">
        <NumberField
          value={value}
          step={integer ? 1 : 0.001}
          precision={integer ? undefined : 4}
          onChange={onChange}
        />
      </span>
    </label>
  )
}