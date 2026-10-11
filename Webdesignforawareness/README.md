# Web Design for Awareness

A browser-based Next.js site with Tailwind CSS styling and Motion scroll
animations.

## Getting started

From this folder, install dependencies and start the development server:

```sh
npm install
npm run dev
```

Open the local URL printed by Next.js in a browser. App Router entry points are
in `app/`. The interactive page lives in `src/App.jsx`, Tailwind theme tokens
are in `src/index.css`, and the orbit/photo styling is in `src/App.css`.

## Production

Build the site and start the production server:

```sh
npm run build
npm start
```

The Earth images are served locally from `public/`. `earth-globe.png` is a
transparent cutout made from NASA's public-domain DSCOVR EPIC Earth photograph
([Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:Earth-DSCOVR-20150706.png)).
The other photographs are sourced from Unsplash:

- `earth-limb.jpg`: `images.unsplash.com/photo-1614730321146-b6fa6a46bcb4`
- `space-orbit.jpg`: `images.unsplash.com/photo-1446776811953-b23d57bd21aa`
