package hub

import (
	"os"

	"github.com/pkg/errors"
)

// DeleteCache deletes the downloaded cache directory for this repository (e.g. ~/.cache/huggingface/hub/models--...).
// If any repository method is called after DeleteCache, files will need to be re-downloaded from HuggingFace Hub.
//
// DeleteCache fails if r is in local-directory mode (r.IsLocal() is true).
func (r *Repo) DeleteCache() error {
	if r.IsLocal() {
		return errors.Errorf("cannot DeleteCache of a local repository %q (local dir %q)", r.ID, r.localDir)
	}
	if r.IsEmbed() {
		return errors.Errorf("cannot DeleteCache of an embedded repository %q", r.ID)
	}

	cacheDir, err := r.repoCacheDir()
	if err != nil {
		return errors.Wrapf(err, "failed to get cache directory for repository %q", r.ID)
	}

	if err := os.RemoveAll(cacheDir); err != nil {
		return errors.Wrapf(err, "failed to delete cache directory %q for repository %q", cacheDir, r.ID)
	}
	return nil
}
