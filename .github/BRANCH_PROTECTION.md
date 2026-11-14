# 🛡️ Política de Protección de Rama (`main`) — GLASTOR® spec-stream

Este documento define la política obligatoria de protección para la rama principal `main` en el repositorio [glastor-core/spec-stream](https://github.com/glastor-core/spec-stream).

---

## 1. Principio Rector

> **Ningún push directo a `main` está permitido.**  
> Todos los cambios sin excepción deben canalizarse a través de un **Pull Request (PR)** verificado que cumpla con los estándares de calidad de GLASTOR® Core.

---

## 2. Reglas Mandatorias de Protección

| Regla | Parámetro | Propósito |
| :--- | :--- | :--- |
| **Require a pull request before merging** | Activado | Impide `git push origin main` directo. |
| **Required approving reviews** | `1` o más | Garantiza revisión por pares (Peer Review). |
| **Require review from Code Owners** | Activado | Exige aprobación de `@glastor-core` o `@glastor-dev` conforme a [`.github/CODEOWNERS`](CODEOWNERS). |
| **Dismiss stale pull request approvals** | Activado | Invalida revisiones previas si se agregan nuevos commits al PR. |
| **Require conversation resolution** | Activado | Todos los hilos y comentarios en el PR deben resolverse antes del merge. |
| **Require status checks to pass** | Activado (Strict) | Exige que la rama esté al día con `main` y que la suite CI pase. |
| **Checks obligatorios requeridos** | 1. `Test Suite (Linux, macOS, Windows)`<br>2. `Clippy & Formatting`<br>3. `Cargo Benchmarks`<br>4. `Workspace Build` | Verifica Rust 1.75+, `cargo clippy`, `cargo fmt` y los 15 tests del workspace. |
| **Require linear history** | Activado | Fomenta historial limpio (`Squash and merge` o `Rebase and merge`), sin merge commits sucios. |
| **Block force pushes** | Activado (`non_fast_forward`) | Previene reescrituras de historia en `main` (`git push --force`). |
| **Block branch deletions** | Activado (`deletion`) | Previene eliminación accidental de la rama `main`. |

---

## 3. Configuración Automatizada (GitHub CLI)

El archivo de configuración de Ruleset oficial se encuentra versionado en [`.github/rulesets/main-protection.json`](rulesets/main-protection.json).

Para aplicar esta protección inmediatamente usando GitHub CLI autenticado con permisos de admin:

```bash
gh api --method POST /repos/glastor-core/spec-stream/rulesets \
  --input .github/rulesets/main-protection.json
```

O para consultar los rulesets activos:

```bash
gh api /repos/glastor-core/spec-stream/rulesets
```

---

## 4. Configuración Manual en GitHub Web

1. Ir a **Settings** -> **Rules** -> **Rulesets** en `github.com/glastor-core/spec-stream`.
2. Clic en **New branch ruleset**.
3. Asignar el nombre: `spec-stream-main-protection`.
4. Enforcement status: **Active**.
5. Target branches: **Include default branch (`main`)**.
6. Marcar:
   - [x] **Restrict deletions**
   - [x] **Block force pushes**
   - [x] **Require a pull request before merging**
   - [x] **Require status checks to pass**
   - [x] **Require linear history**

---

© 2026 Andrés Antonio Cardoso — GLASTOR-DEV / GLASTOR® Core Organization.
