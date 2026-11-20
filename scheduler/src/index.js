/**
 * Trading-bot scheduler: a punctual alarm clock for the GitHub Actions workflows.
 *
 * GitHub's own cron queues scheduled runs and started them 4-7 hours late on this
 * repo, which skipped the whole pre-market window. Runs started through the API
 * (workflow_dispatch) begin immediately, so Cloudflare's Cron Triggers call that API.
 *
 * The bot still checks Alpaca's market calendar itself, so holidays, half days
 * and daylight saving are handled there; this Worker only has to be on time.
 * Cloudflare cron is UTC. See wrangler.jsonc for the New York times.
 */

// Cron expression (must match wrangler.jsonc exactly) -> workflow file to start.
const SCHEDULE = {
  "45 12 * * MON-FRI": "morning.yml",   // main attempt
  "5 13 * * MON-FRI": "morning.yml",    // retry; exits at once if the plan already exists
  "55 16-20 * * MON-FRI": "exits.yml",  // hourly at :55; exits.py acts only just before today's close
  "30 21 * * MON-FRI": "afternoon.yml",
};

export default {
  async scheduled(controller, env, ctx) {
    const workflow = SCHEDULE[controller.cron];
    if (!workflow) {
      throw new Error(`No workflow mapped to cron "${controller.cron}"`);
    }
    await dispatch(workflow, env);
  },
};

async function dispatch(workflow, env) {
  const api = env.GITHUB_API || "https://api.github.com";
  const url = `${api}/repos/${env.GITHUB_REPO}/actions/workflows/${workflow}/dispatches`;

  // One retry covers a brief network or GitHub hiccup; anything else should fail visibly.
  for (let attempt = 1; attempt <= 2; attempt++) {
    let res;
    try {
      res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.GITHUB_TOKEN}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "trading-bot-scheduler",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ref: "main" }),
      });
    } catch (err) {
      if (attempt === 2) throw err;
      continue;
    }
    if (res.status === 204) {
      console.log(`Started ${workflow}`);
      return;
    }
    const detail = await res.text();
    // 5xx: GitHub-side problem, worth one retry. 401/403/404: token or name problem, retrying won't help.
    if (res.status < 500 || attempt === 2) {
      throw new Error(`GitHub refused to start ${workflow}: HTTP ${res.status} ${detail.slice(0, 300)}`);
    }
    await new Promise((r) => setTimeout(r, 5000));
  }
}
