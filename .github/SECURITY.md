# 🛡️ Política de Seguridad (Security Policy) — GLASTOR® spec-stream

En el proyecto **GLASTOR® spec-stream** (`glastor-core/spec-stream`), desarrollado por `GLASTOR-DEV` (división de ingeniería de **GLASTOR® Core**), la seguridad de la memoria, la estabilidad frente a desbordamientos en pools de caché y la integridad de la cadena de suministro son prioridades inquebrantables.

Todo el código de gestión de memoria física, sincronización concurrente con `parking_lot` y operadores de verificación matemática está sujeto a pruebas de concurrencia y validación exhaustiva en CI.

---

## 📦 Versiones Soportadas

Mantenemos actualizaciones de seguridad activas para las versiones estables de `spec-stream`:

| Versión | Estado de Soporte | Tipo de Soporte |
| :--- | :--- | :--- |
| `0.1.x` | :white_check_mark: | Soporte activo y parches inmediatos |
| `< 0.1` | :x: No soportada | Versiones de desarrollo previas |

---

## 🚨 Reporte Responsable de Vulnerabilidades

Si descubres una vulnerabilidad de seguridad o un fallo que comprometa la integridad de `spec-stream` (por ejemplo: fuga de memoria en la tabla de bloques paginados, condición de carrera en operaciones de Copy-on-Write, degradación o bucles infinitos en el muestreo estocástico, o denegación de servicio por agotamiento de recursos en el servidor HTTP), te solicitamos **no abrir un issue público**. En su lugar, activa nuestro protocolo de divulgación coordinada:

### Opción 1: GitHub Private Vulnerability Reporting (Recomendado)

Accede directamente a la herramienta oficial de GitHub:
👉 [Reportar vulnerabilidad privada en spec-stream](https://github.com/glastor-core/spec-stream/security/advisories/new)

### Opción 2: Correo Electrónico Seguro

Envía los detalles técnicos a nuestro equipo de seguridad:
📧 **[ventas@glastor.es](mailto:ventas@glastor.es)** (Email Oficial Web) o **[glastor.info@gmail.com](mailto:glastor.info@gmail.com)** (Personal / Mantenedor)

### Información a Incluir en el Reporte

Para acelerar el triaje y la mitigación, incluye:
- Descripción clara del vector de ataque o vulnerabilidad de memoria.
- Pasos mínimos de reproducción o prueba de concepto (PoC) en Rust.
- Crate afectado (`spec-stream-core`, `spec-stream-cache`, `spec-stream-engine`, `spec-stream-server`).
- Parche o mitigación sugerida (si está disponible).

---

## ⏱️ SLA y Compromiso de Respuesta

- **Recepción y acuse de recibo inicial:** $\le$ 24 horas hábiles.
- **Triaje y evaluación de severidad (CVSS):** $\le$ 48 horas hábiles.
- **Publicación de parche de seguridad:** Prioridad inmediata con aviso público a través de GitHub Security Advisory y crates.io.

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
