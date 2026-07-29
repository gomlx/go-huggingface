# 从本地路径加载模型 —— 可行性分析与实施计划

> 目标：让 `hub.Repo` 除了「从 HuggingFace 远程下载 + 缓存」之外，再支持一种可选的
> 「直接从本地普通目录加载」的模式，使得
> `transformer.LoadModel` / `tokenizers.New` / `safetensors` / `gguf` / `sam2`
> 等上层代码**完全不用改**即可离线运行。
>
> 参考用例：`D:\golang\gomlx-BGE-test\bgetest.go`（BAAI/bge-small-zh-v1.5 句向量编码器）。

---

## 1. 结论

**可行，而且改动面很小。** 建议在 `hub` 包内新增「本地目录模式」，
对外只增加 `hub.NewLocal(dir)` / `Repo.WithLocalDir(dir)` 两个入口，
内部在 4 个方法上分支（`DownloadInfo` / `DownloadFilesCtx` / `FileURL` / `CacheDir`），
**不需要修改 `models/*`、`tokenizers/*` 中的任何一行代码**。

预计工作量：新增 `hub/local.go` 约 150～200 行 + 单元测试约 120 行 + 文档更新。

唯一不支持本地模式的是 `datasets` 包（原因见 [§6.5](#65-datasets-包不在支持范围)）。

---

## 2. 现状分析：上层到底依赖 `*hub.Repo` 的什么

把仓库里所有非 `hub` 包对 `*hub.Repo` 的调用点列出来（`hub` 之外全部调用点）：

| 调用方 | 使用的方法 |
|---|---|
| `models/transformer/model.go:54` | `repo.DownloadFile(name)` |
| `models/safetensors/iter.go:118,134` | `repo.IterFileNames()`、`repo.DownloadFile(name)` |
| `models/gguf/model.go:58` | `repo.IterFileNames()`、`repo.DownloadFile(name)` |
| `models/sam2/model.go:35` | `repo.DownloadFile("config.json")` |
| `tokenizers/tokenizers.go:49,61,79` | `repo.DownloadInfo(false)`、`repo.HasFile(...)`、`repo.DownloadFile(...)` |
| `tokenizers/hftokenizer`、`tokenizers/sentencepiece` | `repo.HasFile(...)`、`repo.DownloadFile(...)` |
| `cmd/hubinfo` | `repo.DownloadInfo`、`repo.Info()`、`repo.IterFileInfos()` |
| 各 example | `repo.DownloadInfo(false)`、`repo.DownloadFile(...)` |
| 错误信息 | 公开字段 `repo.ID` |

也就是说，上层真正的「接口」只有这 7 个：

```
DownloadInfo(force) / Info() / HasFile(name) / IterFileNames() / IterFileInfos()
DownloadFile(s)[Ctx](names...) / 字段 ID
```

其中 `HasFile`、`IterFileNames`、`IterFileInfos` 都是**基于 `r.info.Siblings` 派生**的
（见 `hub/files.go:23-83`），所以真正需要改写的只有两个原语：

1. **`DownloadInfo`** —— 提供 `RepoInfo.Siblings`（文件清单）；
2. **`DownloadFilesCtx`** —— 把「仓库内相对路径」映射成「本地可读的绝对路径」。

只要这两个在本地模式下改成读文件系统，整条链路自然全通。
`safetensors` 用 `os.Open` + `mmap` 打开返回的路径（`iter.go:152-158`），
本地真实文件比缓存里的符号链接更友好，没有额外问题。

---

## 3. 为什么现在「把模型放在本地目录」跑不通

当前 `hub.Repo` 有三处硬绑定到远程：

1. **`DownloadInfo` 必须先拿到 HF API 的 JSON**（`hub/info.go:163-202`）。
   本地目录里没有 `info/main` 这个文件，就一定会发起 `GET {endpoint}/api/models/{id}/revision/main?blobs=true`。
   没有它，`r.info` 为 nil ⇒ `HasFile` 恒为 false、`IterFileNames` 直接 yield error。

2. **`DownloadFilesCtx` 的第一步就是 `r.FileURL(name)`**（`hub/files.go:200`），
   而 `FileURL` → `readCommitHashForRevision()`（`hub/repo.go:196-208`）里有：

   ```go
   forceDownload := !r.revisionHashRefreshed
   err := r.DownloadInfo(forceDownload)   // 第一次必定 forceDownload=true
   ```

   **即使 info 已经完整缓存在磁盘上，进程内第一次取文件也一定会强制联网刷新一次 commit hash。**
   这也意味着「手工伪造一份 HF 缓存目录」这个绕行方案**仍然需要一次联网**，不是真正的离线。

3. **目录布局是 HF 专有的**：`{cacheDir}/models--A--B/{info,blobs,snapshots/{sha}/...}`，
   并且 `snapshots` 下是指向 `blobs/{etag}` 的**符号链接**（`hub/files.go:274,360-376`）。
   而用户手里的目录通常是 `git clone` 或 `huggingface-cli download --local-dir` 得到的**扁平普通目录**：

   ```
   E:/models/bge-small-zh-v1.5/
     ├── config.json
     ├── model.safetensors
     ├── tokenizer.json
     ├── tokenizer_config.json
     ├── modules.json
     └── 1_Pooling/config.json
   ```

   注意示例 `bgetest.go:51` 的 `WithCacheDir("E:/huggingface-models")` **不是**这个意思：
   它改的是「HF 缓存根目录」，仍然按 `models--BAAI--bge-small-zh-v1.5/snapshots/...` 布局，
   仍然会联网。这正是本提案要补上的能力差别。

---

## 4. 方案对比

| 方案 | 做法 | 评价 |
|---|---|---|
| **A. 在 `Repo` 内加本地模式**（推荐） | 加字段 `localDir`，`DownloadInfo`/`DownloadFilesCtx`/`FileURL`/`CacheDir` 分支 | 上层零改动；保持 `*hub.Repo` 具体类型签名不变；实现集中在一个新文件里 |
| B. 抽出接口 `Repo` interface + `LocalRepo` 实现 | 把 `models/*`、`tokenizers/*` 的参数从 `*hub.Repo` 改成接口 | **破坏性 API 变更**，要改 8 个包的公开签名（`LoadModel`、`New`、`TokenizerConstructor`…），收益与 A 相同 |
| C. 不改代码，让用户手工构造 HF 缓存布局 | 手写 `info/main` JSON + `snapshots/{sha}/` | UX 极差，且因为 §3.2 的强制刷新**依然要联网**，不成立 |

**采用方案 A。** 方案 B 可以作为将来（v1.0 大版本）真正需要多后端（S3/OCI/本地）时的演进方向，
届时把 `LocalRepo` 独立出来即可，A 的实现代码可直接复用。

---

## 5. 详细设计

### 5.1 对外 API

新增（全部向后兼容，纯增量）：

```go
// NewLocal 创建一个从本地目录读取的 Repo：不联网、不使用缓存目录。
// dir 是一个包含 config.json / *.safetensors / tokenizer.json 等文件的普通目录，
// 例如 git clone 或 `huggingface-cli download --local-dir` 的产物，
// 也可以直接指向 HF 缓存中的某个 snapshots/{commit-hash} 目录。
func NewLocal(dir string) *Repo

// WithLocalDir 把一个已有的 Repo 切换成本地模式；传入 "" 可切回远程模式。
func (r *Repo) WithLocalDir(dir string) *Repo

// IsLocal 返回该 Repo 是否处于本地模式。
func (r *Repo) IsLocal() bool

// LocalDir 返回本地模式下的根目录（远程模式返回 ""）。
func (r *Repo) LocalDir() string
```

`Repo.ID` 保持公开可写：本地模式下默认取目录的 base name（仅用于日志与错误信息），
需要更可读的名字时用户可自行 `repo.ID = "BAAI/bge-small-zh-v1.5"`。

### 5.2 新增文件 `hub/local.go`

```go
package hub

// Repo 新增字段（写在 hub/repo.go 的 struct 里）：
//   // localDir 非空时，Repo 工作在本地目录模式：不联网，直接从该目录读取文件。
//   localDir string

func NewLocal(dir string) *Repo {
    r := New("")            // 复用默认值（Verbosity 等）
    return r.WithLocalDir(dir)
}

func (r *Repo) WithLocalDir(dir string) *Repo {
    if dir == "" {
        r.localDir = ""
        return r
    }
    resolved, err := files.ReplaceTildeInDir(dir)   // 复用 internal/files，支持 "~/models/..."
    if err != nil {
        log.Printf("Failed to resolve directory for %q: %+v", dir, err)
        resolved = dir
    }
    r.localDir = filepath.Clean(resolved)
    r.info = nil                                    // 失效掉可能存在的远程 info
    r.revisionHashRefreshed = false
    if r.ID == "" {
        r.ID = filepath.Base(r.localDir)
    }
    return r
}

func (r *Repo) IsLocal() bool   { return r.localDir != "" }
func (r *Repo) LocalDir() string { return r.localDir }
```

### 5.3 需要分支的四个方法

| 方法 | 位置 | 本地模式行为 |
|---|---|---|
| `DownloadInfo(force)` | `hub/info.go:163` | 函数开头 `if r.IsLocal() { return r.scanLocalInfo(force) }`，**在任何 `repoCacheDir()` 调用之前**，避免在本地目录里创建 `info/` 子目录 |
| `DownloadFilesCtx(ctx, names...)` | `hub/files.go:131` | 开头改走 `r.localFiles(names...)`：纯路径映射 + 存在性检查，不建 download manager、不建目录、不创建符号链接 |
| `FileURL(name)` | `hub/repo.go:184` | 返回明确错误：`repository %q is in local mode (dir %q): it has no remote URL` |
| `CacheDir()` / `repoCacheDir()` | `hub/repo.go:166,177` | 返回 `localDir` 本身，且**不执行 `os.MkdirAll`** |

`repoSnapshotsDir()`、`readCommitHashForRevision()` 在本地模式下不会再被调用到（调用者只有
`FileURL` 和 `DownloadFilesCtx`），保险起见也各加一个 `IsLocal()` 早退错误分支。

### 5.4 `scanLocalInfo`：用文件系统合成 `RepoInfo`

```go
func (r *Repo) scanLocalInfo(forceRescan bool) error {
    if r.info != nil && !forceRescan {
        return nil
    }
    st, err := os.Stat(r.localDir)
    if err != nil || !st.IsDir() {
        return errors.Wrapf(err, "local model directory %q is not accessible", r.localDir)
    }

    info := &RepoInfo{ID: r.ID, ModelID: r.ID, CommitHash: LocalCommitHash /* "local" */}
    err = filepath.WalkDir(r.localDir, func(p string, d fs.DirEntry, err error) error {
        if err != nil {
            return err
        }
        name := ...  // filepath.ToSlash(rel(r.localDir, p))
        if d.IsDir() {
            if skipLocalDir(name) {          // ".git", ".cache", ".huggingface", ".ipynb_checkpoints"
                return fs.SkipDir
            }
            return nil
        }
        fi, statErr := os.Stat(p)            // 用 Stat 而非 d.Info()，以便跟随符号链接取到真实大小
        if statErr != nil {
            return nil                       // 断链的符号链接：跳过而不是整体失败
        }
        if fi.IsDir() {
            return nil
        }
        info.Siblings = append(info.Siblings, &FileInfo{Name: name, Size: fi.Size()})
        return nil
    })
    if err != nil {
        return errors.Wrapf(err, "while scanning local model directory %q", r.localDir)
    }
    sort.Slice(info.Siblings, func(i, j int) bool { return info.Siblings[i].Name < info.Siblings[j].Name })
    r.info = info
    return nil
}
```

要点：

- **文件名一律用正斜杠**（`filepath.ToSlash`）。上层是按仓库内路径字面比较的
  （`"1_Pooling/config.json"`、`"tokenizer.json"`），Windows 下若写成反斜杠会全部匹配不上。
  这是本方案在 Windows 上最容易踩的坑。
- `FileInfo.BlobID` / `LFS` 留空——它们只在 `cmd/hubinfo` 展示时使用，`hubinfo` 已能处理零值。
- 可选增强：若存在 `config.json`，顺带解析出 `model_type` / `architectures` 填进 `RepoInfo.Config`，
  让 `cmd/hubinfo` 对本地目录也有像样的输出。属于加分项，可放到 Phase 3。

### 5.5 `localFiles`：路径映射

```go
func (r *Repo) localFiles(repoFiles ...string) ([]string, error) {
    paths := make([]string, len(repoFiles))
    for i, name := range repoFiles {
        rel := cleanRelativeFilePath(name)   // 复用 hub/files.go:87，已剥离 ".." 与前导 "/"
        if rel == "." {
            return nil, errors.Errorf("invalid file name %q", name)
        }
        p := filepath.Join(r.localDir, rel)
        if !files.Exists(p) {
            return nil, errors.Errorf("file %q not found in local model directory %q", name, r.localDir)
        }
        paths[i] = p
    }
    return paths, nil
}
```

「文件不存在 ⇒ 返回 error」这个语义与远程模式一致，因此上层的容错逻辑天然复用：
`transformer.LoadModel`（`models/transformer/model.go:53-91`）把任何 `DownloadFile` 错误
都当作「该可选文件不存在」，只有 `config.json` 缺失才硬失败——本地模式下行为完全正确。

---

## 6. 边界情况与已知限制

### 6.1 Windows 路径

见 §5.4：清单里存正斜杠，回填时用 `cleanRelativeFilePath`（内部 `filepath.FromSlash`）转回平台分隔符。
单测需在 Windows 上验证 `1_Pooling/config.json` 这类子目录文件。

### 6.2 目录里含符号链接

允许把 `localDir` 直接指向 HF 缓存的快照目录，例如：

```go
hub.NewLocal("E:/huggingface-models/models--BAAI--bge-small-zh-v1.5/snapshots/<sha>")
```

此时目录内全是指向 `../../blobs/<etag>` 的符号链接。
`os.Stat` 跟随链接可拿到真实大小，`os.Open`/`mmap` 也正常。
断链的符号链接在扫描时静默跳过（不让整个仓库不可用）。

### 6.3 安全性

`cleanRelativeFilePath` 已阻断 `..` 与绝对路径逃逸。
目录内部指向外部的符号链接不做拦截——本地目录属于用户自有资产，
与 `os.Root`（Go 1.24+）的强隔离相比，这里更看重 `mmap` 需要真实路径的实用性。此点在文档中注明即可。

### 6.4 会被忽略的选项

本地模式下 `WithAuth` / `WithEndpoint` / `WithRevision` / `WithCacheDir` / `WithProgressBar`
/ `WithExtraBlobsInfo` / `MaxParallelDownload` 均无意义。
处理方式：不报错，`Verbosity >= 2` 时打一条 debug 日志；在 godoc 中逐条注明「本地模式下忽略」。

### 6.5 `datasets` 包不在支持范围

`datasets.Dataset` 内嵌 `*hub.Repo`，但它的元信息走的是另一套接口
（`datasets/info.go:183,237` 直接请求 dataset-server 的 URL，不经过 `Repo.DownloadInfo`）。
因此本地模式对 `datasets` 无效。
处理方式：在 `datasets` 的构造路径上检测 `repo.IsLocal()` 并返回明确错误
（"datasets do not support local-directory mode"），而不是让用户遇到费解的网络错误。

### 6.6 与「离线模式」的区别（本次不做）

本提案解决的是「模型在一个普通目录里」。
另一个相关但独立的需求是「模型已在标准 HF 缓存里，希望完全不联网」——
它受阻于 §3.2 的 `revisionHashRefreshed` 强制刷新。
将来可用 `WithOffline(true)` 或识别 `HF_HUB_OFFLINE` 环境变量解决，属于独立 issue。

---

## 7. 分阶段实施计划

| 阶段 | 内容 | 产出 | 估时 |
|---|---|---|---|
| **P1 核心** | `hub/repo.go` 加 `localDir` 字段；新增 `hub/local.go`（`NewLocal`/`WithLocalDir`/`IsLocal`/`LocalDir`/`scanLocalInfo`/`localFiles`）；在 `DownloadInfo`、`DownloadFilesCtx`、`FileURL`、`repoCacheDir`/`CacheDir` 上加分支 | 可运行的本地加载 | 0.5 天 |
| **P2 测试** | `hub/local_test.go` 单测 + 与远程结果对拍的集成测试（见 §8） | 测试通过 | 0.5 天 |
| **P3 打磨** | `datasets` 的显式报错；`scanLocalInfo` 解析 `config.json` 填充 `RepoInfo.Config`；`cmd/hubinfo` 增加 `--local-dir` 参数；被忽略选项的日志与 godoc | 体验完善 | 0.5 天 |
| **P4 文档** | `hub/README.md`、`hub/hub.go` 包注释、`docs/CHANGELOG.md`、`examples/BAAI-bge-small-en-v1.5/README.md` 增加本地加载示例 | 文档齐备 | 0.25 天 |

P1 与 P2 可以合成一个 PR；P3、P4 可以拆分。

---

## 8. 测试计划

**单元测试** `hub/local_test.go`（全部离线、用 `t.TempDir()`，可进 CI）：

1. 在临时目录写入 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`1_Pooling/config.json`、
   一个 `.git/objects/xx` 干扰文件；
2. 断言 `IterFileNames()` 返回的名字集合正确、**均为正斜杠**、且**不含 `.git/` 下的文件**；
3. 断言 `IterFileInfos()` 的 `Size` 与写入字节数一致；
4. 断言 `HasFile("1_Pooling/config.json") == true`、`HasFile("nope.json") == false`；
5. 断言 `DownloadFile("config.json")` 返回可读的真实路径；
6. 断言 `DownloadFile("missing.json")` 返回 error 且错误信息里带目录名；
7. 断言 `DownloadFile("../../etc/passwd")` 不会逃出 `localDir`；
8. 断言 `FileURL(...)` 在本地模式下报错；
9. 断言 `NewLocal("/不存在的目录")` 在 `DownloadInfo` 时给出清晰错误；
10. 断言本地模式下扫描过程**没有在 `localDir` 里创建任何新文件/目录**（对比扫描前后的目录快照）——
    防止 `repoCacheDir()` 的 `MkdirAll` 漏了分支。

**集成测试**（需真实模型，打 `testing.Short()` 跳过标记）：

- 用远程 `hub.New("BAAI/bge-small-en-v1.5")` 下载一次，拿到 snapshot 目录；
- 再用 `hub.NewLocal(<snapshot 目录>)` 走一遍 `transformer.LoadModel` + `GetTokenizer` + `SingleSentenceEmbeddingExec`；
- 断言两条路径产出的句向量逐元素一致（可直接复用 `examples/BAAI-bge-small-en-v1.5/similarity_embeddings.txt` 的基准数据）。

---

## 9. 调用方的改动示例

以 `D:\golang\gomlx-BGE-test\bgetest.go` 为例，`NewBGEEncoder` 里只需替换构造 `repo` 的那两行
（`bgetest.go:51-55`）：

```go
// 现在（必须联网；WithCacheDir 只是换了 HF 缓存根目录，仍是 models--*/snapshots/* 布局）
repo := hub.New(ModelName).WithCacheDir("E:/huggingface-models")
if err := repo.DownloadInfo(false); err != nil {
    return nil, err
}
```

```go
// 改造后（完全离线；ModelDir 指向含 config.json 的普通目录）
repo := hub.NewLocal(ModelDir)
repo.ID = ModelName          // 可选：让日志/错误信息更可读
if err := repo.DownloadInfo(false); err != nil {   // 本地模式下等价于「扫描目录」
    return nil, err
}
```

后面的 `transformer.LoadModel(repo)`、`hfModel.GetTokenizer()`、`hfModel.LoadStore(...)`、
`SingleSentenceEmbeddingExec(...)` **一行都不用改**。

再进一步，可以让程序同时支持两种来源：

```go
func newRepo() *hub.Repo {
    if dir := os.Getenv("BGE_MODEL_DIR"); dir != "" {
        return hub.NewLocal(dir)
    }
    return hub.New(ModelName).WithCacheDir("E:/huggingface-models")
}
```

---

## 10. 明确不做的部分

- 不改 `models/*`、`tokenizers/*` 的任何公开签名（不做接口化重构，见方案 B）。
- 不做 `hub.New(x)` 的「自动判断 x 是路径还是模型 ID」——歧义太大（模型 ID 也可能恰好是本地已存在的目录名），
  显式的 `NewLocal` 更安全。
- 不做 HF 标准缓存的离线模式（§6.6），另开 issue。
- 不做本地目录的写入/发布能力（upload），只读。
