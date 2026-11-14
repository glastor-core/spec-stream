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

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
