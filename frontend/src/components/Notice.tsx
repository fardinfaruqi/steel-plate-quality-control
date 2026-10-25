import type { ReactNode } from 'react'

interface NoticeProps {
  variant: 'error' | 'status'
  children: ReactNode
}

export function Notice({ variant, children }: NoticeProps) {
  return (
    <div role={variant === 'error' ? 'alert' : 'status'} className={`notice notice--${variant}`}>
      {children}
    </div>
  )
}