import { useState, type ReactNode } from 'react'

interface ExpanderProps {
  title: string
  defaultOpen?: boolean
  children: ReactNode
}

export function Expander({ title, defaultOpen = false, children }: ExpanderProps) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="panel mb-3 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full cursor-pointer items-center justify-between gap-3 bg-transparent px-5 py-3.5 text-left"
      >
        <span className="font-display text-[0.85rem] font-bold uppercase tracking-[0.12em] text-hot-white [font-stretch:125%]">
          {title}
        </span>
        <svg
          viewBox="0 0 16 16"
          width="12"
          height="12"
          aria-hidden="true"
          className={`shrink-0 text-steel-dim transition-transform duration-200 ${
            open ? 'rotate-90' : ''
          }`}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M6 3l5 5-5 5" />
        </svg>
      </button>
      <div hidden={!open} className="border-t border-hairline px-5 py-4">
        {children}
      </div>
    </div>
  )
}
