// App Router layouts also export route metadata, so component-only fast refresh does not apply.
/* oxlint-disable react/only-export-components */
import '../src/index.css'

export const metadata = {
  title: 'Orbit / Awareness',
  description:
    'A closer look at the human-made debris orbiting Earth and the risks it creates.',
}

export const viewport = {
  themeColor: '#050607',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
