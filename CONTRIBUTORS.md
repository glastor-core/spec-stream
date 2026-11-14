# 🏛️ Gobernanza, Escalabilidad y Sostenibilidad Técnica — GLASTOR® spec-stream

![Contributors](https://img.shields.io/github/contributors/glastor-core/spec-stream)
![Stars](https://img.shields.io/github/stars/glastor-core/spec-stream)
![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)
![Rust 1.75+](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)

`GLASTOR-DEV` (división de ingeniería de **GLASTOR® Core**) concibe **spec-stream** como el motor de inferencia especulativa y gestión de memoria KV-Cache paginada de ultra alto rendimiento y agnóstico de modelo para el ecosistema **Rust** (motores de inferencia LLM, proxies de OpenAI compatibles, servidores edge y pipelines con Candle/ONNX).

Nuestra arquitectura se sustenta en tres pilares inquebrantables:
1. **Rendimiento Extremo y Cero Asignación**: Verificación matemática en paralelo (1.79 μs para $K=4$) y reversión instantánea de propuestas descartadas en tiempo $O(1)$ (249 ns) sin realocación de memoria.
2. **Exactitud Probabilística y Rigor Teórico**: Implementación estricta del muestreo por rechazo estocástico (Leviathan et al., 2023) garantizando matemáticamente que la distribución de salida sea 100% idéntica a la del modelo objetivo.
3. **Agnóstico y Modular**: Workspace desacoplado en 5 crates con traits extensibles (`ModelEngine`), soporte para streaming asíncrono Tokio / Server-Sent Events (SSE) y compatibilidad directa con la API de OpenAI.

---

### 👥 Liderazgo y Dirección de Ingeniería

El desarrollo y dirección estratégica de GLASTOR® spec-stream es coordinado por:

- **Andrés Antonio Cardoso** ([@glastor-dev](https://github.com/glastor-dev))
  - *Lead Software Architect & Project Maintainer*
  - Responsable de la visión arquitectónica, los algoritmos de verificación especulativa y la gestión de memoria Paged KV-Cache para el ecosistema `glastor-core`.

---

### 🚀 Framework de Contribución (Rust-First DX)

Incentivamos una **Developer Experience (DX)** de clase mundial para la comunidad de código abierto:

1. **Compilación en Rust Puro**: Workspace modular organizado jerárquicamente. Todo se valida mediante `cargo build --workspace`, `cargo test --workspace` y `cargo clippy --workspace`.
2. **Seguridad de Memoria y Concurrencia**: Gestión de punteros y bloques físicos protegida mediante `parking_lot::Mutex` y primitivas atómicas seguras con cero condiciones de carrera en Copy-on-Write.
3. **Gobernanza y Versionado Semántico**: Cumplimiento estricto de SemVer 2.0.0 en todas las firmas públicas de los crates.
4. **Protección de Rama Principal (`main`)**: Todos los aportes deben canalizarse mediante Pull Request con revisión aprobada y la suite completa de CI en verde.

Para contribuir con nuevos algoritmos de verificación, soporte de hardware o adaptadores de modelos (Candle, Burn, ONNX), consulta nuestra guía en [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) y abre un Pull Request hacia la rama `main`.

---

### 💎 Ecosistema GLASTOR® Core

| Módulo / Crate | Propósito | Directorio |
| :--- | :--- | :--- |
| **`spec-stream-core`** | Primitivas matemáticas, sampling y verificación de árboles | `crates/spec-stream-core` |
| **`spec-stream-cache`** | Paged KV-Cache, pool físico, prefix caching y rollback $O(1)$ | `crates/spec-stream-cache` |
| **`spec-stream-engine`** | Orquestador asíncrono Tokio, adaptive $K$ y stream pipeline | `crates/spec-stream-engine` |
| **`spec-stream-server`** | Servidor HTTP compatible con OpenAI, SSE y métricas Prometheus | `crates/spec-stream-server` |
| **`spec-stream`** | Crate fachada con preludios ergonómicos y flags de compilación | `crates/spec-stream` |

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
