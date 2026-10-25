// Browser-Oberfläche des Annotators. Keine Abhängigkeiten.
"use strict";

const zustand = {
  bilder: [],
  klassen: [],
  aktuell: null,
  boxes: [],
  gewaehlt: null,
  klasse: null,
  ziehen: null,
  bild: null,
};

const el = (id) => document.getElementById(id);
const buehne = el("buehne");
const ctx = buehne.getContext("2d");
const STATUS_TEXT = {
  new: "neu",
  auto_ok: "Vorschlag",
  needs_review: "Prüfung",
  labeled: "fertig",
};

// Rückgängig: nur Änderungen dieser Sitzung.
const verlauf = [];

function merke(eintrag) {
  verlauf.push(eintrag);
  if (verlauf.length > 100) verlauf.shift();
}

let toastZeit = null;
function zeigeToast(text, fehler = false) {
  const kasten = el("toast");
  kasten.textContent = text;
  kasten.classList.remove("leer");
  kasten.classList.toggle("fehler", fehler);
  if (toastZeit) clearTimeout(toastZeit);
  toastZeit = setTimeout(() => kasten.classList.add("leer"), 6000);
}

async function anfrage(pfad, optionen = {}) {
  const antwort = await fetch(pfad, {
    headers: { "Content-Type": "application/json" },
    ...optionen,
  });
  if (!antwort.ok) {
    let text = antwort.statusText;
    try {
      const koerper = await antwort.json();
      text = koerper.detail || text;
    } catch (fehler) {
      /* kein JSON */
    }
    throw new Error(text);
  }
  return antwort.status === 204 ? null : antwort.json();
}

async function ladeKlassen() {
  zustand.klassen = await anfrage("/api/classes");
  const behaelter = el("klassen");
  behaelter.innerHTML = "";
  zustand.klassen.forEach((klasse) => {
    const knopf = document.createElement("div");
    knopf.className = "klasse";
    knopf.dataset.id = String(klasse.id);
    knopf.innerHTML = `<span class="farbe" style="background:${klasse.color}"></span>${klasse.name}`;
    knopf.addEventListener("click", () => klasseWaehlen(klasse.id));
    behaelter.appendChild(knopf);
  });
  if (zustand.klassen.length && zustand.klasse === null) {
    klasseWaehlen(zustand.klassen[0].id);
  }
}

function klasseWaehlen(id) {
  zustand.klasse = id;
  document.querySelectorAll(".klasse").forEach((knopf) => {
    knopf.classList.toggle("gewaehlt", Number(knopf.dataset.id) === id);
  });
  if (zustand.gewaehlt !== null) {
    const box = zustand.boxes.find((eintrag) => eintrag.id === zustand.gewaehlt);
    if (box && box.class_id !== id) {
      merke({ art: "patch", id: box.id, vorher: { ...box } });
      anfrage(`/api/boxes/${box.id}`, {
        method: "PATCH",
        body: JSON.stringify({ class_id: id }),
      }).then(() => ladeBild(zustand.aktuell));
    }
  }
  zeichne();
}

async function ladeListe() {
  const filter = el("filter").value;
  const pfad = filter ? `/api/images?status=${encodeURIComponent(filter)}` : "/api/images";
  zustand.bilder = await anfrage(pfad);
  const liste = el("liste");
  liste.innerHTML = "";
  zustand.bilder.forEach((bild) => {
    const eintrag = document.createElement("div");
    eintrag.className = "eintrag" + (zustand.aktuell === bild.id ? " aktiv" : "");
    eintrag.innerHTML = `
      <span class="punkt ${bild.status}" title="${STATUS_TEXT[bild.status] || bild.status}"></span>
      <span class="name" title="${bild.path}">${bild.name}</span>
      <span class="anzahl">${bild.boxes}</span>`;
    eintrag.addEventListener("click", () => ladeBild(bild.id));
    liste.appendChild(eintrag);
  });
  await ladeZusammenfassung();
}

async function ladeZusammenfassung() {
  const daten = await anfrage("/api/summary");
  const teile = Object.entries(daten.counts).map(
    ([status, anzahl]) => `${STATUS_TEXT[status] || status}: ${anzahl}`,
  );
  el("zaehlung").textContent = `Bestand: ${teile.join(" · ")}`;
  const banner = el("banner");
  if (daten.review_queue.length > 0) {
    banner.classList.remove("leer");
    banner.textContent =
      `${daten.review_queue.length} Bild(er) brauchen einen Menschen: ` +
      daten.review_queue
        .slice(0, 5)
        .map((eintrag) => eintrag.name)
        .join(", ") +
      (daten.review_queue.length > 5 ? " …" : "");
  } else {
    banner.classList.add("leer");
    banner.textContent = "";
  }
}

async function ladeBild(id) {
  const daten = await anfrage(`/api/images/${id}`);
  zustand.aktuell = id;
  zustand.boxes = daten.boxes;
  zustand.gewaehlt = null;
  const bild = new Image();
  bild.onload = () => {
    zustand.bild = bild;
    buehne.width = bild.naturalWidth;
    buehne.height = bild.naturalHeight;
    buehne.hidden = false;
    el("leer").hidden = true;
    zeichne();
  };
  bild.src = `/api/images/${id}/file?t=${Date.now()}`;
  document.querySelectorAll(".eintrag").forEach((eintrag, index) => {
    eintrag.classList.toggle("aktiv", zustand.bilder[index] && zustand.bilder[index].id === id);
  });
}

function skalierung() {
  return buehne.width / buehne.clientWidth || 1;
}

function mausPunkt(ereignis) {
  const rechteck = buehne.getBoundingClientRect();
  const faktor = buehne.width / rechteck.width;
  return {
    x: (ereignis.clientX - rechteck.left) * faktor,
    y: (ereignis.clientY - rechteck.top) * faktor,
  };
}

function farbeFuer(classId) {
  const gefunden = zustand.klassen.find((klasse) => klasse.id === classId);
  return gefunden ? gefunden.color : "#4f9dff";
}

function zeichne() {
  if (!zustand.bild) return;
  ctx.clearRect(0, 0, buehne.width, buehne.height);
  ctx.drawImage(zustand.bild, 0, 0);
  const breite = buehne.width;
  const hoehe = buehne.height;
  zustand.boxes.forEach((box) => {
    const x = (box.cx - box.w / 2) * breite;
    const y = (box.cy - box.h / 2) * hoehe;
    const w = box.w * breite;
    const h = box.h * hoehe;
    ctx.lineWidth = Math.max(2, breite / 400);
    ctx.strokeStyle = farbeFuer(box.class_id);
    if (box.state === "proposed") ctx.setLineDash([8, 6]);
    else ctx.setLineDash([]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    const klasse = zustand.klassen.find((eintrag) => eintrag.id === box.class_id);
    ctx.fillStyle = farbeFuer(box.class_id);
    const text = klasse ? klasse.name : `Klasse ${box.class_id}`;
    ctx.font = `${Math.max(12, breite / 60)}px system-ui, sans-serif`;
    ctx.fillText(text, x + 4, Math.max(14, y - 5));
    if (box.id === zustand.gewaehlt) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      const griff = Math.max(6, breite / 120);
      [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([px, py]) => {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(px - griff / 2, py - griff / 2, griff, griff);
      });
    }
  });
  if (zustand.ziehen && zustand.ziehen.art === "neu") {
    const { start, ende } = zustand.ziehen;
    ctx.strokeStyle = "#4f9dff";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(start.x, start.y, ende.x - start.x, ende.y - start.y);
    ctx.setLineDash([]);
  }
}

function boxUnterPunkt(punkt) {
  const breite = buehne.width;
  const hoehe = buehne.height;
  for (let i = zustand.boxes.length - 1; i >= 0; i -= 1) {
    const box = zustand.boxes[i];
    const x = (box.cx - box.w / 2) * breite;
    const y = (box.cy - box.h / 2) * hoehe;
    if (
      punkt.x >= x &&
      punkt.x <= x + box.w * breite &&
      punkt.y >= y &&
      punkt.y <= y + box.h * hoehe
    ) {
      return box;
    }
  }
  return null;
}

function eckeUnterPunkt(box, punkt) {
  const breite = buehne.width;
  const hoehe = buehne.height;
  const x = (box.cx - box.w / 2) * breite;
  const y = (box.cy - box.h / 2) * hoehe;
  const w = box.w * breite;
  const h = box.h * hoehe;
  const griff = Math.max(10, breite / 60);
  const ecken = [
    ["nw", x, y],
    ["no", x + w, y],
    ["sw", x, y + h],
    ["so", x + w, y + h],
  ];
  for (const [name, px, py] of ecken) {
    if (Math.abs(punkt.x - px) <= griff && Math.abs(punkt.y - py) <= griff) return name;
  }
  return null;
}

buehne.addEventListener("mousedown", (ereignis) => {
  if (!zustand.bild) return;
  const punkt = mausPunkt(ereignis);
  const gewaehlt = zustand.boxes.find((box) => box.id === zustand.gewaehlt);
  const ecke = gewaehlt ? eckeUnterPunkt(gewaehlt, punkt) : null;
  if (ecke) {
    zustand.ziehen = { art: "groesse", ecke, box: { ...gewaehlt }, start: punkt };
    return;
  }
  const getroffen = boxUnterPunkt(punkt);
  if (getroffen) {
    zustand.gewaehlt = getroffen.id;
    zustand.ziehen = { art: "bewegen", box: { ...getroffen }, start: punkt };
    zeichne();
    return;
  }
  zustand.ziehen = { art: "neu", start: punkt, ende: punkt };
});

buehne.addEventListener("mousemove", (ereignis) => {
  if (!zustand.ziehen) return;
  const punkt = mausPunkt(ereignis);
  if (zustand.ziehen.art === "neu") {
    zustand.ziehen.ende = punkt;
    zeichne();
    return;
  }
  if (zustand.ziehen.art === "bewegen") {
    const dx = (punkt.x - zustand.ziehen.start.x) / buehne.width;
    const dy = (punkt.y - zustand.ziehen.start.y) / buehne.height;
    const box = zustand.ziehen.box;
    const verschoben = {
      ...box,
      cx: Math.min(0.999, Math.max(0.001, box.cx + dx)),
      cy: Math.min(0.999, Math.max(0.001, box.cy + dy)),
    };
    ersetzeLokal(verschoben);
    return;
  }
  if (zustand.ziehen.art === "groesse") {
    const box = zustand.ziehen.box;
    const links = (box.cx - box.w / 2) * buehne.width;
    const oben = (box.cy - box.h / 2) * buehne.height;
    const rechts = (box.cx + box.w / 2) * buehne.width;
    const unten = (box.cy + box.h / 2) * buehne.height;
    let x0 = links;
    let y0 = oben;
    let x1 = rechts;
    let y1 = unten;
    if (zustand.ziehen.ecke.includes("w")) x0 = punkt.x;
    if (zustand.ziehen.ecke.includes("o")) x1 = punkt.x;
    if (zustand.ziehen.ecke.includes("n")) y0 = punkt.y;
    if (zustand.ziehen.ecke.includes("s")) y1 = punkt.y;
    const kleinX = Math.min(x0, x1);
    const kleinY = Math.min(y0, y1);
    const neu = {
      ...box,
      cx: (kleinX + Math.max(x0, x1)) / 2 / buehne.width,
      cy: (kleinY + Math.max(y0, y1)) / 2 / buehne.height,
      w: Math.max(2, Math.abs(x1 - x0)) / buehne.width,
      h: Math.max(2, Math.abs(y1 - y0)) / buehne.height,
    };
    ersetzeLokal(neu);
  }
});

function ersetzeLokal(box) {
  zustand.boxes = zustand.boxes.map((eintrag) => (eintrag.id === box.id ? box : eintrag));
  zeichne();
}

buehne.addEventListener("mouseup", async () => {
  const zug = zustand.ziehen;
  zustand.ziehen = null;
  if (!zug) return;
  if (zug.art === "neu") {
    const x0 = Math.min(zug.start.x, zug.ende.x);
    const y0 = Math.min(zug.start.y, zug.ende.y);
    const breite = Math.abs(zug.ende.x - zug.start.x);
    const hoehe = Math.abs(zug.ende.y - zug.start.y);
    if (breite < 4 || hoehe < 4) {
      zeichne();
      return;
    }
    const box = {
      class_id: zustand.klasse || zustand.klassen[0].id,
      cx: (x0 + breite / 2) / buehne.width,
      cy: (y0 + hoehe / 2) / buehne.height,
      w: breite / buehne.width,
      h: hoehe / buehne.height,
    };
    const angelegt = await anfrage(`/api/images/${zustand.aktuell}/boxes`, {
      method: "POST",
      body: JSON.stringify(box),
    });
    zustand.boxes.push(angelegt);
    zustand.gewaehlt = angelegt.id;
    merke({ art: "create", id: angelegt.id });
    zeichne();
    await ladeListe();
    return;
  }
  const geaendert = zustand.boxes.find((eintrag) => eintrag.id === zug.box.id);
  if (!geaendert) return;
  const unveraendert =
    geaendert.cx === zug.box.cx &&
    geaendert.cy === zug.box.cy &&
    geaendert.w === zug.box.w &&
    geaendert.h === zug.box.h;
  if (unveraendert) return;
  merke({
    art: "patch",
    id: geaendert.id,
    vorher: { cx: zug.box.cx, cy: zug.box.cy, w: zug.box.w, h: zug.box.h, class_id: zug.box.class_id },
  });
  await anfrage(`/api/boxes/${geaendert.id}`, {
    method: "PATCH",
    body: JSON.stringify({ cx: geaendert.cx, cy: geaendert.cy, w: geaendert.w, h: geaendert.h }),
  });
});

document.addEventListener("keydown", async (ereignis) => {
  if (ereignis.target instanceof HTMLInputElement || ereignis.target instanceof HTMLSelectElement) {
    return;
  }
  const taste = ereignis.key;
  if (taste >= "1" && taste <= "9") {
    const index = Number(taste) - 1;
    if (zustand.klassen[index]) klasseWaehlen(zustand.klassen[index].id);
    return;
  }
  if (taste === "Entf" || taste === "Delete") {
    if (zustand.gewaehlt === null) return;
    ereignis.preventDefault();
    const box = zustand.boxes.find((eintrag) => eintrag.id === zustand.gewaehlt);
    if (box) merke({ art: "delete", id: box.id, box: { ...box } });
    await anfrage(`/api/boxes/${zustand.gewaehlt}`, { method: "DELETE" });
    zustand.boxes = zustand.boxes.filter((eintrag) => eintrag.id !== zustand.gewaehlt);
    zustand.gewaehlt = null;
    zeichne();
    await ladeListe();
    return;
  }
  if ((ereignis.ctrlKey || ereignis.metaKey) && (taste === "z" || taste === "Z")) {
    ereignis.preventDefault();
    await rueckgaengigMachen();
    return;
  }
  if (taste === " " || taste === "Leertaste") {
    ereignis.preventDefault();
    await akzeptieren();
    return;
  }
  if (taste === "a" || taste === "A" || taste === "ArrowLeft") {
    await schritt(-1);
  }
  if (taste === "d" || taste === "D" || taste === "ArrowRight") {
    await schritt(1);
  }
  if (taste === "r" || taste === "R") {
    await ladeListe();
  }
});

async function schritt(richtung) {
  if (!zustand.bilder.length) return;
  const index = zustand.bilder.findIndex((bild) => bild.id === zustand.aktuell);
  const neu = Math.min(zustand.bilder.length - 1, Math.max(0, index + richtung));
  if (neu === index) return;
  await ladeBild(zustand.bilder[neu].id);
}

async function akzeptieren() {
  if (zustand.aktuell === null) return;
  await anfrage(`/api/images/${zustand.aktuell}/accept`, { method: "POST" });
  await ladeBild(zustand.aktuell);
  await ladeListe();
}

async function rueckgaengigMachen() {
  const eintrag = verlauf.pop();
  if (!eintrag) {
    zeigeToast("Nichts rückgängig zu machen.");
    return;
  }
  try {
    if (eintrag.art === "create") {
      await anfrage(`/api/boxes/${eintrag.id}`, { method: "DELETE" });
      zustand.boxes = zustand.boxes.filter((box) => box.id !== eintrag.id);
      zustand.gewaehlt = null;
    } else if (eintrag.art === "patch") {
      const werte = {
        cx: eintrag.vorher.cx,
        cy: eintrag.vorher.cy,
        w: eintrag.vorher.w,
        h: eintrag.vorher.h,
        class_id: eintrag.vorher.class_id,
      };
      await anfrage(`/api/boxes/${eintrag.id}`, {
        method: "PATCH",
        body: JSON.stringify(werte),
      });
      zustand.boxes = zustand.boxes.map((box) =>
        box.id === eintrag.id ? { ...box, ...werte } : box,
      );
    } else if (eintrag.art === "delete") {
      const angelegt = await anfrage(`/api/images/${zustand.aktuell}/boxes`, {
        method: "POST",
        body: JSON.stringify({
          class_id: eintrag.box.class_id,
          cx: eintrag.box.cx,
          cy: eintrag.box.cy,
          w: eintrag.box.w,
          h: eintrag.box.h,
        }),
      });
      zustand.boxes.push(angelegt);
      zustand.gewaehlt = angelegt.id;
    }
    zeichne();
    await ladeListe();
    zeigeToast("Rückgängig gemacht.");
  } catch (fehler) {
    zeigeToast(`Rückgängig fehlgeschlagen: ${fehler.message}`, true);
  }
}

async function importieren() {
  const ordner = el("importpfad").value.trim();
  if (!ordner) {
    zeigeToast("Bitte einen Ordner angeben.", true);
    return;
  }
  try {
    const daten = await anfrage("/api/import", {
      method: "POST",
      body: JSON.stringify({
        folder: ordner,
        recursive: el("rekursiv").checked,
        labels: el("mitlabels").checked,
      }),
    });
    const labelTeil = daten.labels
      ? `, ${daten.labels.boxes} Labels aus ${daten.labels.files} Dateien`
      : "";
    zeigeToast(
      `${daten.added} Bild(er) importiert, ${daten.duplicates} Duplikate, ` +
        `${daten.skipped} übersprungen${labelTeil}.`,
    );
    await ladeListe();
  } catch (fehler) {
    zeigeToast(`Import fehlgeschlagen: ${fehler.message}`, true);
  }
}

async function exportieren() {
  const ziel = el("exportpfad").value.trim() || "./datensatz";
  try {
    const daten = await anfrage("/api/export", {
      method: "POST",
      body: JSON.stringify({
        out: ziel,
        format: el("exportformat").value,
        force: el("exportforce").checked,
      }),
    });
    zeigeToast(
      `Export fertig: ${daten.train} Bilder Training, ${daten.val} Bilder Prüfung → ${daten.out_dir}`,
    );
  } catch (fehler) {
    zeigeToast(`Export fehlgeschlagen: ${fehler.message}`, true);
  }
}

el("filter").addEventListener("change", ladeListe);
el("neu").addEventListener("click", ladeListe);
el("akzeptieren").addEventListener("click", akzeptieren);
el("rueckgaengig").addEventListener("click", rueckgaengigMachen);
el("importieren").addEventListener("click", importieren);
el("exportieren").addEventListener("click", exportieren);
window.addEventListener("resize", zeichne);

(async function start() {
  await ladeKlassen();
  await ladeListe();
})();
