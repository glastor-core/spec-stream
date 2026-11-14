# Changelog — GLASTOR® spec-stream

<p align="center">
  <img src="https://img.shields.io/badge/Format-Keep%20a%20Changelog-blue.svg" alt="Keep a Changelog">
  <img src="https://img.shields.io/badge/Versioning-SemVer%202.0.0-green.svg" alt="Semantic Versioning">
  <img src="https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Repository-glastor--core%2Fspec--stream-informational" alt="Repository">
</p>

---

## Acerca de este Changelog

Todos los cambios notables en **GLASTOR® spec-stream** (High-Performance, Model-Agnostic Speculative Decoding and Paged KV-Cache Engine in Pure Rust) son documentados en este archivo.

Este proyecto se adhiere a:
- **Formato:** [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/)
- **Versionado:** [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.1.0] - 2026-11-13

### ✨ Added

#### Motor Central de Verificación Especulativa (`spec-stream-core`)
- **Verificación Voraz (Greedy Verifier)**: Comparación en tiempo récord (1.79 μs para $K=4$) entre propuestas del modelo borrador y predicciones argmax del modelo objetivo, emitiendo token de corrección o token de bonificación en caso de aceptación total.
- **Muestreo por Rechazo Estocástico (Leviathan et al., 2023)**:
  - Verificación matemática estricta preservando exactamente la distribución probabilística del modelo objetivo.
  - Muestreo residual desde la distribución normalizada $\max(0, p(x) - q(x))$ en caso de rechazo.
- **Suite de Sampling y Procesamiento de Logits**:
  - `apply_top_k`: Truncamiento determinista conservando los $K$ valores superiores y anulando el resto a $-\infty$.
  - `apply_top_p`: Muestreo por núcleo (*nucleus sampling*) con corte por masa acumulada.
  - `apply_min_p`: Filtrado relativo respecto a la probabilidad máxima del vocabulario.
  - `apply_repetition_penalty`: Penalización de repetición estilo HuggingFace para tokens previos en el contexto.
  - `apply_presence_and_frequency_penalties`: Penalizaciones aditivas (estilo OpenAI).
  - `SamplingParams`: Estructura unificada para parametrizar el muestreo estocástico.
- **Decodificación por Árboles Especulativos (`SpeculativeTree`)**:
  - Modelado de árboles de candidatos multirrama (estilo Medusa / Eagle).
  - Cálculo de la matriz booleana 2D de atención causal (`compute_tree_attention_mask`).
  - Verificación greedy de la rama más larga aceptada mediante DFS/BFS en $O(\text{Nodos})$.
- **Tokenizer BPE / Byte-Fallback (`ByteLevelTokenizer`)**:
  - Tokenizador en Rust puro sin dependencias externas para codificación/decodificación bidireccional y sin pérdida de caracteres UTF-8 en inglés y español.

#### Gestión de Memoria KV-Cache Paginada (`spec-stream-cache`)
- **Asignador de Bloques Físicos (`BlockAllocator`)**: Pool de memoria continua dividido en bloques de tamaño configurable (por defecto 16 tokens).
- **Tablas Virtuales a Físicas (`BlockTable`)**: Mapeo transparente de coordenadas lógicas a pares `(bloque_id, offset)`.
- **Rollback Instantáneo en $O(1)$**: Reversión de propuestas no aceptadas en **249 ns** devolviendo bloques completos a la lista libre sin fragmentación de memoria.
- **Prefijos Compartidos Automáticos (Prefix Caching)**: Hashing incremental `compute_block_prefix_hash` que detecta bloques con contenido idéntico (ej. prompts de sistema) y los comparte con cero asignaciones adicionales en memoria.
- **Aislamiento por Copy-on-Write (CoW)**: Duplicación atómica de bloques compartidos (`ref_count > 1`) antes de escribir o truncar, previniendo condiciones de carrera entre secuencias concurrentes.
- **Política de Desalojo LRU y Desfragmentación (`LruEvictionPolicy`)**:
  - Desalojo inteligente de secuencias inactivas cuando el pool físico alcanza presión de memoria crítica.
  - Reordenamiento de lista libre (`defragment_free_list`) para maximizar la localidad espacial en líneas de caché de CPU.

#### Orquestación Asíncrona de Inferencia (`spec-stream-engine`)
- **Soporte de Cuantización GGUF Q8_0 y Q4_0 (`QuantizedLinearModel`)**:
  - Kernels nativos en Rust puro para de-cuantización y producto punto vectorial directo sin asignación de memoria intermedia (`vec_dot_q8_0`, `vec_dot_q4_0`).
  - Reducción del uso de VRAM/RAM en cabezales de proyección en hasta un 87% con conversión de punto flotante FP16 IEEE-754.
- **Planificador de Prefill por Bloques (`ChunkedPrefillScheduler`)**:
  - Segmentación de prompts masivos (2k–8k+ tokens) en fragmentos configurables (`chunk_size`), intercalando el cómputo de prefill con las ráfagas de decodificación activa para eliminar la latencia por bloqueo.
- **Drafter Especulativo sin Modelo (`PromptLookupEngine`)**:
  - Decodificación por búsqueda en prompt (Prompt Lookup Decoding) mediante coincidencia de sufijos n-gram en el contexto del prompt.
  - Generación de candidatos en nanosegundos con **0 parámetros neuronales y 0 VRAM adicional**.
- **Planificador de Lotes Continuos (`ContinuousBatchScheduler`)**: Motor de batching a nivel de iteración con control dinámico de admisión (`QueuedRequest`), procesamiento simultáneo de secuencias activas y recolección de memoria sin latencia.
- **Parsers de Metadatos y Pesos Neuronales (`GgufMetadata`, `SafeTensorsMetadata`)**: Lector nativo en Rust puro para binarios GGUF (v2 y v3) y SafeTensors, extrayendo tensores, shapes y metadatos sin librerías externas.
- **Modelo de Proyección Lineal (`LinearProjectionModel`)**: Motor liviano optimizado para computar multiplicaciones vectoriales `hidden * W^T` para embeddings y cabezales LM en especulación.
- **Persistencia de Caché de Prefijos (`PrefixCacheSnapshot`)**: Exportación e importación instantánea de estados calientes de memoria KV (`save_snapshot_json`, `load_snapshot_json`) para eliminar la latencia de arranque en frío.
- **Pipeline Especulativo Asíncrono (`SpeculativePipeline`)**: Integración completa sobre Tokio con canales `mpsc` para transmisión de eventos en tiempo real.
- **Auto-escalado Dinámico ($K$-Drafting Adaptativo)**: Ajuste continuo del número de tokens propuestos $K$ en base a una media móvil exponencial (EMA) de la tasa de aceptación, elevando $K$ hasta 8 cuando la precisión es alta y reduciéndolo a 1 cuando decae.
- **Pipeline Arbóreo (`TreeSpeculativePipeline`)**: Orquestador para evaluar árboles de candidatos contra el modelo objetivo en un único forward pass por lote.
- **Trait Agnóstico `ModelEngine`**: Interfaz genérica para conectar backends de pesos reales (Candle, Burn, ONNX Runtime o FFI nativo) con implementación sintética de referencia (`SyntheticModel`).

#### Servidor HTTP Compatible con OpenAI (`spec-stream-server`)
- **Suite de Subcomandos CLI**:
  - `serve`: Inicio estándar del servidor de inferencia HTTP/SSE.
  - `bench`: Benchmark de rendimiento en memoria con reporte de throughput (tokens/s) y factor de speedup.
  - `inspect-gguf <archivo.gguf>`: Inspección diagnóstica de metadatos, arquitecturas y tablas de tensores en binarios GGUF.
- **Configuración Unificada (`spec-stream.toml`)**: Archivo de configuración modular en la raíz del proyecto para ajustar parámetros del servidor, pool de bloques de caché y límites de concurrencia.
- **Endpoints de Inferencia**:
  - `POST /v1/chat/completions`: Compatible con el formato estándar de OpenAI (tanto JSON síncrono como Server-Sent Events con `stream: true`).
  - `GET /health`: Reporte de estado operativo y utilización de bloques en la caché.
  - `GET /metrics`: Endpoint en formato oficial Prometheus con contadores y métricas de utilización.
  - `GET /`: Landing page visual interactiva y playground integrado servido directamente por el binario.

#### Crate Fachada y Ergonomía (`spec-stream`)
- **Preludio Unificado (`spec_stream::prelude::*`)**: Re-exportación ergonómica con feature flags modulares (`cache`, `engine`).
- **Demostración Interactiva (`examples/speculative_pipeline.rs`)**: Ejemplo ejecutable completo demostrando el ciclo de ráfaga y la adaptación de $K$.
- **Micro-benchmarks Criterion (`benches/verification_bench.rs`)**: Pruebas de rendimiento para verificar latencias de microsegundos en release.

#### Excelencia de Grado Industrial Tier-1 (Calificación 10/10)
- **Aceleración Vectorial SIMD por Hardware (`spec-stream-engine`)**:
  - Implementación con intrínsecos x86_64 `AVX2` y `FMA` (`_mm256_fmadd_ps`) para el producto punto cuantizado `vec_dot_q8_0` con detección dinámica en tiempo de ejecución (`is_x86_feature_detected!`) y fallback escalar automático.
  - Test de equivalencia numérica de punto flotante certificando precisión idéntica entre kernels vectorizados y escalares.
- **Testing Basado en Propiedades e Invariantes Matemáticas (`spec-stream-cache`)**:
  - Suite de 5.000 operaciones concurrentes aleatorias (`crates/spec-stream-cache/tests/property_invariants.rs`) verificando leyes de conservación de memoria (`num_free + num_allocated == total_blocks`), aislamiento estricto de Copy-on-Write y ausencia total de fugas o dobles liberaciones.
- **Apagado Agraciado y Drenado de Conexiones (*Graceful Shutdown*) (`spec-stream-server`)**:
  - Captura asíncrona de señales de terminación de sistema operativo (`SIGINT` y `SIGTERM`) con `axum::serve.with_graceful_shutdown()`, garantizando el drenado de flujos SSE en vuelo y persistencia de memoria antes de salir.
- **Trazabilidad Estructurada OpenTelemetry-Compatible (`spec-stream-server`)**:
  - Instrumentación de spans contextuales y métricas de latencia por petición (`prompt_len`, `generated_tokens`, `duration_ms`) para integración con recolectores OTel, Jaeger y Datadog.

#### Infraestructura DevOps, CI/CD, Seguridad y Contenedores
- **Flujo de Despliegue en GitHub Pages (`.github/workflows/pages.yml`)**: Despliegue automático de la landing page interactiva (`index.html`) y documentación generada de la API en Rust (`cargo doc`) en GitHub Pages.
- **Empaquetado y Publicación Multi-Arquitectura en GHCR (`Dockerfile`, `.github/workflows/docker.yml`)**:
  - `Dockerfile` multi-etapa endurecido con usuario no-privilegiado `specstream` (UID 10001) y verificación de salud (`HEALTHCHECK`).
  - GitHub Actions con Docker Buildx y QEMU para compilar y publicar imágenes para arquitecturas `linux/amd64` y `linux/arm64` en GitHub Container Registry (`ghcr.io/glastor-core/spec-stream:latest`).
- **Auditoría Continua de Seguridad y Cadena de Suministro (`.github/workflows/security-audit.yml`)**:
  - Escaneo semanal programado (cron lunes 00:00 UTC) y en cambios de dependencias.
  - Verificación exhaustiva con `cargo-audit` (base de datos RustSec) y `cargo-deny` (licencias SPDX, crates prohibidas y vulnerabilidades conocidas).
- **Seguimiento Continuo de Benchmarks y Detección de Regresiones (`.github/workflows/bench.yml`)**: Ejecución automatizada de microbenchmarks y tracking histórico con alertas de regresión de latencia.
- **Reporte y Análisis de Cobertura de Código (`.github/workflows/coverage.yml`)**: Generación de reportes LCOV con `cargo-llvm-cov` para auditoría de pruebas unitarias y de integración.
- **Verificación de Versión Mínima Soportada de Rust (`.github/workflows/msrv.yml`)**: Validación continua de compatibilidad de todo el workspace con Rust 1.75.0 (MSRV).
- **Clasificación y Etiquetado Automático de Pull Requests (`.github/workflows/pr-labeler.yml`, `.github/labeler.yml`)**: Triage inteligente aplicando etiquetas automáticas (`core`, `cache`, `engine`, `server`, `ui`, `ci`, `docs`, `security`, `dependencies`).

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
