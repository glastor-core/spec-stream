## 📋 Descripción del Cambio

<!-- Explica de forma concisa qué problema resuelve este Pull Request o qué funcionalidad optimiza en GLASTOR® spec-stream. -->

## 🎯 Tipo de Cambio

- [ ] 🐛 Bug fix (corrección de error sin breaking changes en la verificación o en el KV-cache)
- [ ] ✨ Nueva característica (ej: nuevo algoritmo de sampling, backend de modelo o endpoint en el servidor)
- [ ] 💥 Breaking change (cambio que altera interfaces públicas de traits o estructuras centrales)
- [ ] ⚡ Optimización de rendimiento (mejora de micro-benchmarks o reducción de latencia en la verificación)
- [ ] 📝 Documentación (actualización de README, index.html o doctests)
- [ ] 🔧 Mantenimiento / CI / Tooling (refactorización, Clippy, workflows de GitHub Actions)

## 🔗 Issues Relacionados

<!-- Vincula los issues afectados, por ejemplo: Fixes #123, Closes #456 -->

- Fixes #

## 🧪 Pruebas Realizadas

<!-- Marca las verificaciones ejecutadas localmente antes de enviar el PR: -->

- [ ] Formateo estricto: `cargo fmt --all --check`
- [ ] Linter Clippy sin advertencias: `cargo clippy --workspace --all-targets`
- [ ] Suite de pruebas completa: `cargo test --workspace`
- [ ] Validación de ejemplos: `cargo run --example speculative_pipeline`
- [ ] Verificación de benchmarks: `cargo bench -p spec-stream`
- [ ] Validación de la landing page / showcase: apertura en navegador local de `index.html`

## 🛡️ Checklist de Protección de Rama (`main`)

- [ ] Mi rama parte de la versión más reciente de `main`.
- [ ] Los mensajes de commit cumplen con [COMMIT_CONVENTION.md](./COMMIT_CONVENTION.md).
- [ ] He actualizado el archivo `CHANGELOG.md` si corresponde.
- [ ] No se introducen dependencias de producción externas innecesarias.

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
