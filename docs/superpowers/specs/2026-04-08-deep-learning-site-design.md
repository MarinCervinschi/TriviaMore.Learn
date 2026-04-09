# Study Hub — Design Spec & Roadmap

> Documento di riferimento per il progetto. Contiene design, roadmap, stato di avanzamento e decisioni prese.
> Aggiornato man mano che il progetto evolve.

---

## Overview

Sito statico multi-corso per presentare materiale di studio universitario. Struttura modulare: ogni corso ha la sua sezione con capitoli dedicati. Il primo corso e' Deep Learning (15 capitoli), ma l'architettura supporta l'aggiunta di nuovi corsi senza modifiche strutturali.

**Obiettivo**: materiale di studio chiaro, navigabile, cercabile. Il contenuto e' il protagonista.

---

## Decisioni prese

| Decisione | Scelta | Motivazione |
|-----------|--------|-------------|
| Framework | Astro + MDX | Content-first, output statico, componenti dentro markdown |
| Formule | KaTeX (rehype-katex) | Rendering LaTeX veloce, integrazione nativa con MDX |
| Ricerca | Pagefind | Full-text search client-side, zero backend, indicizza al build |
| Diagrammi | Mermaid inline in MDX | Diagrammi solo dove aggiungono valore, il contenuto e' gia' ricco. No standalone HTML. |
| Lingua | Inglese | I markdown originali sono in inglese |
| Deploy | Da decidere (Vercel / GitHub Pages / VPS) | Flessibile, sito statico deployabile ovunque |
| Design | Pulito e semplice | Focus sul contenuto, niente complessita' inutile |
| Struttura | Multi-corso modulare | Content Collections per corso, aggiungere un corso = aggiungere una cartella |

---

## Architettura

```
study-hub/
├── astro.config.mjs
├── package.json
├── src/
│   ├── layouts/
│   │   ├── CourseLayout.astro         # Layout condiviso per le pagine corso
│   │   └── ChapterLayout.astro       # Layout condiviso per tutti i capitoli
│   ├── components/
│   │   ├── Sidebar.astro              # Navigazione laterale (capitoli del corso corrente)
│   │   ├── TableOfContents.astro      # TOC con heading del capitolo corrente
│   │   ├── CourseCard.astro           # Card per ogni corso nella homepage
│   │   ├── MathBlock.astro            # Wrapper KaTeX per formule
│   │   ├── Diagram.astro              # Contenitore per diagrammi visual-explainer
│   │   └── SearchBar.astro            # Integrazione Pagefind
│   ├── content/
│   │   ├── config.ts                  # Schema Astro Content Collections (per corso)
│   │   └── courses/
│   │       └── deep-learning/
│   │           ├── _meta.json         # Metadata corso (titolo, descrizione, ordine capitoli)
│   │           ├── 01-deep-neural-networks.mdx
│   │           ├── 02-convolutional-neural-networks.mdx
│   │           ├── ...
│   │           └── 15-reinforcement-learning.mdx
│   │       # └── machine-learning/    # Futuro: aggiungere una cartella = aggiungere un corso
│   ├── pages/
│   │   ├── index.astro                # Homepage con lista corsi
│   │   └── [course]/
│   │       ├── index.astro            # Pagina overview del corso
│   │       └── [slug].astro           # Pagina capitolo
│   └── styles/
│       └── global.css                 # Stili globali, tipografia, KaTeX
├── public/
│   └── images/
│       └── deep-learning/
│           ├── 01-deep-neural-networks/
│           ├── ...
│           └── 15-reinforcement-learning/
```

**Aggiungere un nuovo corso**: creare una cartella in `content/courses/<nome-corso>/` con `_meta.json` e i file MDX. Le immagini vanno in `public/images/<nome-corso>/`. Nessuna modifica a componenti o routing.

**Flusso dati**: Markdown originali -> script di migrazione -> MDX puliti -> Astro Content Collections -> HTML statico

**Navigazione**: Homepage (lista corsi) -> Pagina corso (overview + lista capitoli) -> Capitolo (sidebar + TOC + prev/next)

**Ricerca**: Pagefind indicizza l'output HTML al build time, search bar nell'header (cerca in tutti i corsi)

**URL structure**:
- `/` — homepage con tutti i corsi
- `/deep-learning/` — overview del corso
- `/deep-learning/01-deep-neural-networks` — capitolo specifico

---

## Materiale sorgente

15 capitoli markdown con 218 immagini PNG totali, esportati da Notion.

| # | Capitolo | Immagini | File sorgente |
|---|----------|----------|---------------|
| 01 | Deep Neural Networks | 16 | Deep Neural Network 27eb575baf6880ca97d1c6045988f0af.md |
| 02 | Convolutional Neural Networks | 9 | Convutional Neural Network 28cb575baf68805da48ac316833eca36.md |
| 03 | Recurrent Neural Networks | 9 | Recurrent Neural Networks 28cb575baf688043a65cec41825d6d4b.md |
| 04 | Transformers | 11 | Trasformers 29cb575baf68803f88dbe077e3599b3e.md |
| 05 | Graph Neural Networks | 18 | Graph Neural Networks 2a1b575baf6880b3ba40d4fdf73f255e.md |
| 06 | Autoencoders | 10 | Autoencoders 29ab575baf6880d6a154f4941293d3b7.md |
| 07 | Generative Models | 12 | Generative models 2acb575baf6880948163f774a12f6b8c.md |
| 08 | Discrete AI | 8 | Discrete AI 2afb575baf688015be87fb8b2dffd148.md |
| 09 | Uncertainty | 13 | Uncertainty 2b6b575baf688096a285f743fca88671.md |
| 10 | Self-Supervised Learning | 12 | Self-Supervised Learning 2c6b575baf6880238341e1c61d1b05ca.md |
| 11 | Transfer Learning | 8 | Transfer Learning 2c6b575baf68807194a1efb17f6aa8dc.md |
| 12 | Parameter-Efficient Fine-Tuning | 9 | Parameter-Efficient Fine-Tuning 2c6b575baf68807da1d3ea6e1b6fd231.md |
| 13 | Continual Learning | 22 | Continual Learning 2c4b575baf6880d2b4acfa1a563342d2.md |
| 14 | Federated Learning | 6 | Federated Learning 31fb575baf6880c3b2c6d9b067aedf93.md |
| 15 | Reinforcement Learning | 13 | Reinforcement Learning 321b575baf6880b9a886e1874d1a0501.md |

---

## Roadmap

### Fase 0 — Scaffolding e infrastruttura
> Preparare tutto il necessario per poi lavorare capitolo per capitolo

- [x] Inizializzare progetto Astro con MDX, KaTeX, Pagefind
- [x] Creare CourseLayout.astro e ChapterLayout.astro (sidebar + TOC + prev/next)
- [x] Creare CourseCard.astro per la homepage
- [x] Configurare Content Collections multi-corso (schema con _meta.json per corso)
- [x] Creare homepage (index.astro) con lista corsi
- [x] Creare pagina overview corso ([course]/index.astro)
- [x] Creare routing dinamico capitoli ([course]/[slug].astro)
- [x] Creare script di migrazione: markdown Notion -> MDX puliti (fix nomi, path immagini, frontmatter)
- [x] Migrare immagini in public/images/deep-learning/ organizzate per capitolo
- [x] Verificare: sito funzionante con tutti i capitoli, navigazione e ricerca attivi

**Stato**: `COMPLETATA`

---

### Fase 1 — Primi 5 capitoli (fondamenti)
> I capitoli fondamentali delle architetture neurali

Per ogni capitolo: revisione contenuto, fix formule LaTeX, verifica immagini, Mermaid inline dove utile, tabella riassuntiva a fine capitolo.

- [x] 01 — Deep Neural Networks
- [x] 02 — Convolutional Neural Networks
- [ ] 03 — Recurrent Neural Networks
- [ ] 04 — Transformers
- [ ] 05 — Graph Neural Networks
- [ ] Verifica: tutti e 5 navigabili, formule renderizzate, immagini visibili, ricerca funzionante

**Stato**: `IN CORSO`

---

### Fase 2 — Capitoli 6-10 (modelli generativi e apprendimento)
> Modelli generativi, incertezza, apprendimento non supervisionato

- [ ] 06 — Autoencoders
- [ ] 07 — Generative Models
- [ ] 08 — Discrete AI
- [ ] 09 — Uncertainty
- [ ] 10 — Self-Supervised Learning
- [ ] Verifica: 10 capitoli totali completati e coerenti

**Stato**: `DA FARE`

---

### Fase 3 — Capitoli 11-15 e rifinitura
> Tecniche avanzate di training e deployment + rifinitura finale

- [ ] 11 — Transfer Learning
- [ ] 12 — Parameter-Efficient Fine-Tuning
- [ ] 13 — Continual Learning
- [ ] 14 — Federated Learning
- [ ] 15 — Reinforcement Learning
- [ ] Rifinitura: responsive, performance, meta tags, favicon
- [ ] Deploy finale
- [ ] Verifica: sito completo, tutti i capitoli, pronto per la produzione

**Stato**: `DA FARE`

---

## Come usare questo file

1. **Prima di ogni fase**: preparare un piano di implementazione specifico
2. **Durante il lavoro**: spuntare le checkbox man mano che si completano i task
3. **Aggiornare lo stato** della fase corrente (`DA FARE` -> `IN CORSO` -> `COMPLETATA`)
4. **Decisioni nuove**: aggiungerle alla tabella "Decisioni prese"
5. **Problemi/note**: aggiungerli in fondo al documento

---

## Note e problemi aperti

- Contenuti migrati as-is da Notion; le fasi 1-3 servono per revisione e rifinitura di ogni capitolo
- Sorgenti Notion originali rimossi dal repo dopo migrazione (2026-04-08)
- Package manager: pnpm
- Tailwind CSS v4 (via @tailwindcss/vite, non @astrojs/tailwind)
- Astro v6 Content Layer API con render() standalone
- Approccio diagrammi rivisto (2026-04-09): niente visual-explainer/HTML standalone, solo Mermaid inline dove serve. Focus su contenuto pulito + tabella riassuntiva a fine capitolo.
- Per ogni capitolo: fix contenuto → buona formattazione → Mermaid se utile → tabella Key Takeaways
