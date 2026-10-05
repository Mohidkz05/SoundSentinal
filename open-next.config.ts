// OpenNext for Cloudflare. Every page here is client-rendered or prerendered
// and nothing uses ISR, so the read-only static-assets cache is enough: no R2
// bucket (which needs a payment method on the account) and no KV.
import { defineCloudflareConfig } from "@opennextjs/cloudflare";
import staticAssetsIncrementalCache from "@opennextjs/cloudflare/overrides/incremental-cache/static-assets-incremental-cache";

export default defineCloudflareConfig({
	incrementalCache: staticAssetsIncrementalCache,
});
