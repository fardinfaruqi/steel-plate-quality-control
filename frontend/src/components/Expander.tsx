import { useId, useState, type ReactNode } from 'react'

interface ExpanderProps {
  title: string
  defaultOpen?: boolean
  children: ReactNode
}

export function Expander({ title, defaultOpen = false, children }: ExpanderProps) {
  const [open, setOpen] = useState(defaultOpen)
  const bodyId = useId()

  return (
    <div className="disclosure">
      <button
        type="button"
        className="disclosure__toggle"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((o) => !o)}
      >
        <span>{title}</span>
        <svg
          className="disclosure__mark"
          viewBox="0 0 16 16"
          width="12"
          height="12"
          aria-hidden="true"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M6 3l5 5-5 5" />
        </svg>
      </button>
      <div className="disclosure__body" id={bodyId} hidden={!open}>
        {children}
      </div>
    </div>
  )
}