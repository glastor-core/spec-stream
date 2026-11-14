# 🌌 Guía de Contribución y Estándares de Ingeniería — GLASTOR® spec-stream

Gracias por sumarte a la misión de construir y optimizar **GLASTOR® spec-stream** dentro del ecosistema de `GLASTOR-DEV` (división de ingeniería de **GLASTOR® Core**).

`spec-stream` es el motor de inferencia especulativa y memoria KV-cache paginada de ultra alto rendimiento para LLMs en Rust, diseñado para superar el cuello de botella de ancho de banda de memoria en arquitecturas Transformer.

---

## 🛡️ Política de Protección de Rama (`main`)

Para garantizar la máxima estabilidad y la calidad del código en producción:
- **La rama `main` está protegida**: No se permiten envíos directos (`git push origin main`) ni modificaciones forzadas (`--force`).
- **Todo cambio debe ingresar mediante Pull Request (PR)**:
  - Debe crearse una rama temática (`feat/nombre`, `fix/nombre`, `perf/nombre`).
  - La suite automatizada de GitHub Actions (tests unitarios/integración en Linux, macOS y Windows, Clippy y formato) debe finalizar en estado verde (`[PASS]`).
  - Se requiere al menos una revisión aprobada por los mantenedores ([CODEOWNERS](./CODEOWNERS)).

---

## 💻 Setup del Entorno de Desarrollo

### Requisitos Previos

- **Rust**: Versión estable 1.75+ (Edición 2021).
- **Git**: 2.30+.
- **Zero Heavy Toolchains**: Cero dependencias externas requeridas para compilación por defecto.

### 1. Clonación del Repositorio

```bash
git clone https://github.com/glastor-core/spec-stream.git
cd spec-stream
```

### 2. Comandos de Trabajo y Verificación

```bash
# Compilar todo el workspace modular
cargo build --workspace

# Ejecutar las 15 pruebas unitarias y de integración
cargo test --workspace

# Verificar cumplimiento de formato oficial
cargo fmt --all --check

# Ejecutar linter estricto de Clippy
cargo clippy --workspace --all-targets

# Ejecutar la demostración de pipeline especulativo
cargo run --example speculative_pipeline

# Ejecutar micro-benchmarks con Criterion
cargo bench -p spec-stream

# Iniciar el servidor HTTP OpenAI en localhost:8080
cargo run -p spec-stream-server
```

---

## 📐 Estándares de Código y Arquitectura

1. **Modularidad Estricta**:
   - `spec-stream-core`: Cero dependencias de I/O de red o HTTP. Solo matemática pura, sampling y estructuras de datos.
   - `spec-stream-cache`: Solo gestión de memoria paginada, tablas virtuales y Copy-on-Write.
   - `spec-stream-engine`: Orquestación asíncrona sobre Tokio sin acoplamiento a HTTP.
   - `spec-stream-server`: Capa de transporte Axum, SSE y compatibilidad con OpenAI.
2. **Cero Alojamientos Innecesarios**:
   - Aprovechar `SmallVec` para listas de candidatos y buffers de tokens de tamaño acotado.
   - Reutilizar bloques físicos en `BlockAllocator` mediante hashing de prefijos compartidos.
3. **Compatibilidad con Ecosistemas de Inferencia**:
   - Nuevos modelos deben implementar el trait `ModelEngine` de forma asíncrona.

---

## 🤝 Proceso de Envío de Pull Requests

1. Crea una rama descriptiva a partir de la versión más reciente de `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/candle-backend
   ```
2. Realiza tus cambios asegurando que las pruebas existentes y nuevas pasen al 100%.
3. Escribe mensajes de commit bajo el estándar [Conventional Commits](./COMMIT_CONVENTION.md).
4. Abre el Pull Request en GitHub seleccionando como destino la rama `main` y completa el formulario de [PULL_REQUEST_TEMPLATE.md](./PULL_REQUEST_TEMPLATE.md).

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
