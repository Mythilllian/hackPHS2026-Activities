'use client'

import Image from 'next/image'
import { useRef } from 'react'
import { motion, useReducedMotion, useScroll, useTransform } from 'motion/react'
import './App.css'

const debris = [
  { className: 'debris-panel', label: 'fragmented solar panel' },
  { className: 'debris-chip', label: 'metal fragment' },
  { className: 'debris-fragment', label: 'small debris fragment' },
  { className: 'debris-satellite', label: 'defunct satellite' },
  { className: 'debris-spark', label: 'distant fragment' },
  { className: 'debris-chip debris-chip-small', label: 'metal shard' },
]

function Earth() {
  return (
    <div className="earth" aria-hidden="true">
      <Image
        src="/earth-globe.png"
        alt=""
        fill
        priority
        sizes="(max-width: 640px) 45vw, 280px"
        className="earth-image"
      />
      <span className="earth-edge" aria-hidden="true" />
    </div>
  )
}

function OrbitScene({ rotation, reverseRotation }) {
  return (
    <div className="orbit-scene" role="img" aria-label="Space debris orbiting a photographic view of Earth">
      <div className="orbit-glow" aria-hidden="true" />
      <Earth />
      <motion.div className="orbit-track orbit-track-outer" style={{ rotate: rotation }}>
        {debris.slice(0, 3).map((item) => (
          <span
            key={item.className}
            className={`debris-piece ${item.className}`}
            title={item.label}
          />
        ))}
      </motion.div>
      <motion.div className="orbit-track orbit-track-inner" style={{ rotate: reverseRotation }}>
        {debris.slice(3).map((item) => (
          <span
            key={item.className}
            className={`debris-piece ${item.className}`}
            title={item.label}
          />
        ))}
      </motion.div>
      <div className="orbit-label orbit-label-top" aria-hidden="true">
        <span className="status-dot" />
        LOW EARTH ORBIT
      </div>
      <div className="orbit-label orbit-label-bottom" aria-hidden="true">
        ORBITAL DEBRIS / LIVE VIEW
      </div>
    </div>
  )
}

function App() {
  const hookSection = useRef(null)
  const prefersReducedMotion = useReducedMotion()
  const { scrollYProgress } = useScroll({
    target: hookSection,
    offset: ['start start', 'end end'],
  })
  const rotation = useTransform(scrollYProgress, [0, 1], [0, prefersReducedMotion ? 45 : 590])
  const reverseRotation = useTransform(scrollYProgress, [0, 1], [0, prefersReducedMotion ? -30 : -390])
  const progressWidth = useTransform(scrollYProgress, [0, 1], ['0%', '100%'])
  const progressLabel = useTransform(
    scrollYProgress,
    (value) => `${Math.round(value * 100).toString().padStart(2, '0')}%`,
  )

  return (
    <main id="top" className="relative min-h-screen overflow-clip bg-void text-mist">
      <motion.div
        className="scroll-progress"
        style={{ scaleX: scrollYProgress }}
        aria-hidden="true"
      />
      <div className="starfield" aria-hidden="true" />

      <header className="site-header">
        <a className="brand" href="#top" aria-label="Orbit Awareness home">
          <span className="brand-mark" aria-hidden="true">
            O
          </span>
          <span>ORBIT <span className="brand-slash">/</span> AWARENESS</span>
        </a>
        <nav className="site-nav" aria-label="Main navigation">
          <a href="#hook">THE ISSUE</a>
          <a className="nav-cta" href="#story">
            THE BIGGER PICTURE <span aria-hidden="true">↗</span>
          </a>
        </nav>
      </header>

      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow">
            <span className="status-dot" />
            AWARENESS STARTS ABOVE US
          </p>
          <h1 id="hero-title">
            The sky is not
            <br />
            <span>as empty</span> as it looks.
          </h1>
          <p className="hero-description">
            Pieces of human-made debris circle our planet at incredible speed.
            Scroll to see the space we all share.
          </p>
          <a className="text-link" href="#hook">
            <span className="link-arrow" aria-hidden="true">↓</span>
            SCROLL TO EXPLORE
          </a>
        </div>

        <figure className="hero-earth">
          <motion.div
            className="hero-earth-stage"
            role="img"
            aria-label="The Earth, photographed by NASA's DSCOVR satellite"
            tabIndex={0}
            whileHover={
              prefersReducedMotion ? {} : { scale: 1.045, rotateY: 12, y: -8 }
            }
            whileFocus={
              prefersReducedMotion ? {} : { scale: 1.045, rotateY: 12, y: -8 }
            }
            transition={{ type: 'spring', stiffness: 135, damping: 18 }}
          >
            <span className="hero-earth-aura" aria-hidden="true" />
            <span className="hero-orbit-ring hero-orbit-ring-wide" aria-hidden="true" />
            <span className="hero-orbit-ring hero-orbit-ring-tall" aria-hidden="true" />
            <Image
              src="/earth-globe.png"
              alt=""
              fill
              priority
              sizes="(max-width: 760px) 90vw, 48vw"
              className="hero-earth-image"
            />
          </motion.div>
          <figcaption className="photo-caption">
            <span>EARTH / DSCOVR EPIC</span>
            <span>NASA PUBLIC DOMAIN</span>
          </figcaption>
        </figure>

        <span className="hero-index" aria-hidden="true">AWARENESS / 001</span>
      </section>

      <section
        id="hook"
        ref={hookSection}
        className="hook-section"
        aria-labelledby="hook-title"
      >
        <div className="hook-sticky">
          <div className="hook-layout">
            <div className="hook-copy">
              <p className="eyebrow">
                <span className="status-dot" />
                THE HOOK
              </p>
              <h2 id="hook-title">
                Millions of fragments of metal
                <br />
                <span>orbit Earth at Mach 23.</span>
              </h2>
              <div className="hook-narrative">
                <p className="hook-lede">
                When we think of threats to modern infrastructure, we usually
                think of earthquakes, cyberattacks, or extreme weather.
                We don’t think about the sky falling.
                </p>
                <p className="hook-description">
                But in 2024, a metal cylinder from the International Space
                Station survived atmospheric reentry and smashed directly
                through a Florida home, exposing a terrifying reality: our
                unmanaged cosmic trash is turning the atmosphere into a
                firing range, putting both orbital technology and
                terrestrial life in the literal line of fire.
                </p>
              </div>
              <div className="scroll-instruction">
                <span className="progress-track">
                  <motion.span style={{ width: progressWidth }} />
                </span>
                SCROLL TO MOVE THE ORBIT
              </div>
            </div>
            <div className="orbit-panel">
              <OrbitScene
                rotation={rotation}
                reverseRotation={reverseRotation}
              />
            </div>
          </div>
          <div className="hook-bottomline" aria-hidden="true">
            <span>EARTH IS NOT A LANDFILL.</span>
            <span>
              SCROLL PROGRESS / <motion.span>{progressLabel}</motion.span>
            </span>
          </div>
        </div>
      </section>

      <section id="story" className="story-section">
        <div className="section-label">
          <span className="eyebrow">A CLOSER LOOK</span>
          <span className="section-index">01 / 02</span>
        </div>
        <div className="story-grid">
          <div className="story-image">
            <Image
              src="/space-orbit.jpg"
              alt="The blue Earth floating in the darkness of space"
              fill
              sizes="(max-width: 700px) 100vw, 50vw"
              className="cover-image"
            />
            <div className="story-image-caption">
              <span>THE VIEW WE SHARE</span>
              <span>EARTH / 01</span>
            </div>
          </div>
          <div className="story-copy">
            <span className="story-number">01</span>
            <h2>One shared orbit. A story still unfolding.</h2>
            <p>
              This space is ready for your facts, your perspective, and the
              actions you want people to take. We, as humans, are responsible for the trash that we leave behind. It is our duty as an intelligent species to acknowledge the wellbeing of our home.
            </p>
            <a className="text-link" href="#top">
              <span className="link-arrow" aria-hidden="true">↑</span>
              BACK TO THE TOP
            </a>
          </div>
        </div>
      </section>

      <section className="action-section" aria-labelledby="action-title">
        <p className="eyebrow">TAKE THE NEXT STEP</p>
        <h2 id="action-title">Be a part of the cause</h2>
        <a className="action-button" href="#story">
          See how you can get involved
          <span aria-hidden="true">↗</span>
        </a>
      </section>

      <footer className="site-footer">
        <span>ORBIT <span className="brand-slash">/</span> AWARENESS</span>
        <span>A starting point for a story worth sharing.</span>
        <a href="#top" aria-label="Back to top">↑</a>
      </footer>
    </main>
  )
}

export default App
