# Search Redesign — Global Modal + Chapter Search

## Contesto

La ricerca Pagefind era implementata come input inline nell'header, ma risultava rotta (import ESM sbagliato) e una volta fixata il dropdown dei risultati era troppo stretto e si sovrapponeva al contenuto. Invece di patchare ulteriormente il dropdown, ridisegniamo la ricerca con due componenti distinti che coprono due use-case diversi.

## Componenti

### 1. Ricerca Globale — SearchModal

**Trigger:**
- Bottone compatto nell'header di tutte le pagine: icona lente + testo "Search..." + badge `Cmd+K`
- Sostituisce l'attuale SearchBar inline
- Shortcut `Cmd+K` (macOS) / `Ctrl+K` (Windows/Linux) apre il modal da qualsiasi pagina

**Modal:**
- Overlay scuro semi-trasparente (`bg-black/50`) su tutta la pagina
- Box centrato, larghezza max `600px`, border-radius `16px`, shadow forte
- Input di ricerca grande in alto con icona lente e placeholder "Search all chapters..."
- Area risultati sotto l'input, max-height `60vh` con scroll
- Ogni risultato mostra: titolo capitolo (link), excerpt con termine evidenziato
- Thumbnails nascosti (come nel fix precedente)

**Interazione:**
- Click su overlay o `Escape` chiude il modal
- Frecce `Up`/`Down` per navigare tra i risultati
- `Enter` per andare al risultato selezionato
- Focus trap: Tab resta dentro il modal quando aperto

**Pagefind:**
- Usa Pagefind JS API direttamente (`pagefind.search()`) invece di PagefindUI
- Questo da controllo completo sul rendering dei risultati
- CSS di pagefind-ui.css non serve piu
- L'import di `/pagefind/pagefind.js` resta con try/catch per dev mode

**Presente su:** tutte le pagine (index.astro, CourseLayout, ChapterLayout)

### 2. Ricerca Capitolo — ChapterSearch

**Posizione:** barra compatta sopra il contenuto dell'articolo in ChapterLayout, sotto il titolo `<h1>`

**UI:**
- Input piccolo con icona lente, placeholder "Search in this chapter..."
- A destra dell'input: conteggio match ("3 of 12") e bottoni prev/next (frecce)
- Bottone X per chiudere e rimuovere gli highlight

**Funzionamento:**
- Cerca nel testo dell'elemento `<article>` corrente
- Usa TreeWalker API per trovare nodi di testo che contengono il termine
- Wrappa le occorrenze in `<mark class="search-highlight">` con classe attiva per il match corrente
- Debounce di 200ms sulla digitazione
- Prev/Next scrolla al match precedente/successivo con `scrollIntoView({ behavior: 'smooth', block: 'center' })`

**Stile highlight:**
- Match generico: `background: #fef08a` (giallo chiaro)
- Match attivo/corrente: `background: #f97316; color: white` (arancione)

**Solo su:** ChapterLayout

## File da creare/modificare

| File | Azione | Descrizione |
|------|--------|-------------|
| `src/components/SearchModal.astro` | Creare | Modal ricerca globale con Pagefind API |
| `src/components/ChapterSearch.astro` | Creare | Ricerca inline nel capitolo corrente |
| `src/components/SearchTrigger.astro` | Creare | Bottone trigger per il modal nell'header |
| `src/components/SearchBar.astro` | Eliminare | Sostituito da SearchModal + SearchTrigger |
| `src/layouts/ChapterLayout.astro` | Modificare | Sostituire SearchBar con SearchTrigger + SearchModal, aggiungere ChapterSearch |
| `src/layouts/CourseLayout.astro` | Modificare | Sostituire SearchBar con SearchTrigger + SearchModal |
| `src/pages/index.astro` | Modificare | Sostituire SearchBar con SearchTrigger + SearchModal |
| `src/styles/global.css` | Modificare | Aggiungere stili per highlight e modal |

## Accessibilita

- Modal: `role="dialog"`, `aria-modal="true"`, `aria-label="Search"`
- Focus trap nel modal (Tab non esce)
- Input con `aria-label` descrittivo
- Risultati navigabili con `aria-activedescendant`
- ChapterSearch: conteggio match annunciato con `aria-live="polite"`
- Tutti i bottoni con `aria-label`

## Verifica

1. `pnpm build && pnpm preview`
2. Aprire homepage — click su "Search..." o Cmd+K apre il modal
3. Cercare un termine — risultati appaiono con excerpt
4. Navigare con frecce e Enter — va alla pagina corretta
5. Escape chiude il modal
6. Aprire un capitolo — barra "Search in this chapter..." visibile
7. Cercare un termine — occorrenze evidenziate in giallo, match corrente in arancione
8. Prev/Next naviga tra i match, conteggio aggiornato
9. X chiude la ricerca e rimuove highlight
