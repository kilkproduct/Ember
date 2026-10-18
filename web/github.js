/** Read/write one private GitHub JSON (the Ember brain). Token stays on-device. */

function headers(token) {
  return {
    Accept: "application/vnd.github+json",
    Authorization: `Bearer ${token}`,
    "X-GitHub-Api-Version": "2022-11-28",
  };
}

function apiUrl(owner, repo, path) {
  return `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}/contents/${path
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;
}

export async function githubHead(owner, repo, path, token) {
  const res = await fetch(apiUrl(owner, repo, path), { headers: headers(token) });
  if (res.status === 404) return { sha: null, json: null };
  if (!res.ok) throw new Error(`GitHub ${res.status}`);
  const data = await res.json();
  return { sha: data.sha, json: data };
}

export async function githubPull(owner, repo, path, token) {
  const { sha, json } = await githubHead(owner, repo, path, token);
  if (!sha || !json?.content) return { sha: null, brain: null };
  const raw = atob(json.content.replace(/\n/g, ""));
  const text = decodeURIComponent(escape(raw));
  return { sha, brain: JSON.parse(text) };
}

export async function githubPush(owner, repo, path, token, brain, sha) {
  const content = btoa(unescape(encodeURIComponent(JSON.stringify(brain))));
  const body = {
    message: `ember brain ${brain.name || ""} step ${brain.step || 0}`.trim(),
    content,
    sha: sha || undefined,
  };
  const res = await fetch(apiUrl(owner, repo, path), {
    method: "PUT",
    headers: { ...headers(token), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`GitHub push ${res.status} ${(await res.text()).slice(0, 160)}`);
  const data = await res.json();
  return data.content?.sha;
}
