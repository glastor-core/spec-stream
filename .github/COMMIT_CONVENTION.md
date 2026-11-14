# 🏛️ Convención de Mensajes de Commit (Conventional Commits) — GLASTOR® spec-stream

Para escalar el desarrollo y garantizar una **Software Supply Chain** transparente, este repositorio implementa una convención estricta basada en el estándar [Conventional Commits 1.0.0](https://www.conventionalcommits.org/).

Esta práctica impulsa la automatización de versiones con SemVer y la generación consistente de changelogs.

---

## ⚙️ Estructura del Mensaje

```text
<tipo>(<scope>): <asunto en minúsculas y modo imperativo>

[cuerpo descriptivo opcional]

[pie de página / referencias opcional]
```

### 1. Tipos de Commit Permitidos

| Tipo | Propósito | Impacto en Release |
| :--- | :--- | :---: |
| `feat` | Nueva funcionalidad, verificador, sampling o endpoint | Minor (`v0.X.0`) |
| `fix` | Corrección de bug en verificación, rollback o caché | Patch (`v0.0.X`) |
| `perf` | Optimización de latencia en verifiers o asignación de memoria | Patch (`v0.0.X`) |
| `refactor` | Reestructuración interna sin alterar comportamiento público | Ninguno |
| `docs` | Actualización de documentación, index.html o README | Ninguno |
| `test` | Nuevas pruebas unitarias, integradas o de concurrencia | Ninguno |
| `ci` | Ajustes en GitHub Actions, workflows o automatizaciones | Ninguno |
| `chore` | Tareas de mantenimiento, dependencias o metadatos Cargo | Ninguno |

### 2. Scopes Oficiales de spec-stream

- `core`: Verificadores matemáticos, logits, sampling y árboles (`crates/spec-stream-core/`).
- `cache`: Paged KV-Cache, asignador de bloques, prefix caching y CoW (`crates/spec-stream-cache/`).
- `engine`: Pipeline especulativo, adaptive $K$, tree pipeline y streams (`crates/spec-stream-engine/`).
- `server`: Servidor HTTP Axum, SSE, handlers OpenAI y Prometheus (`crates/spec-stream-server/`).
- `facade`: Crate fachada, feature flags y preludios (`crates/spec-stream/`).
- `ui`: Landing page interactiva y simulador visual (`index.html`).
- `bench`: Benchmarks de rendimiento Criterion (`benches/`).
- `deps`: Actualizaciones de dependencias en `Cargo.toml`.

---

## 💡 Ejemplos Válidos

```text
feat(core): implement nucleus top-p and min-p sampling algorithms
feat(cache): add copy-on-write isolation for shared prefix blocks
fix(engine): prevent underflow in adaptive K calculation when draft is rejected
perf(cache): optimize physical block rollback to 249 ns in O(1)
docs(ui): enhance interactive tree decoding visualizer in index.html
ci(actions): add Windows and macOS testing matrix to GitHub workflow
```

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
