//! Factory for creating router instances

use super::{
    http::{openai_router::OpenAIRouter, router::Router, vllm_pd_router::VllmPDRouter},
    RouterTrait,
};
use crate::config::{PolicyConfig, RoutingMode};
use crate::kv_events::pool::KVEventPoolConfig;
use crate::kv_events::KVEventPool;
use crate::kv_index::{run_kv_index_updater, EventCache, KVBlockIndex};
use crate::policies::PolicyFactory;
use crate::policies::{CacheAwarePolicy, SMetricPolicy};
use crate::server::AppContext;
use crate::tokenizer::{create_tokenizer_async, traits::Encoder};
use std::sync::Arc;

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        match &ctx.router_config.mode {
            RoutingMode::Regular { worker_urls } => {
                Self::create_regular_router(worker_urls, ctx).await
            }
            RoutingMode::VllmPrefillDecode {
                prefill_urls,
                decode_urls,
                prefill_policy,
                decode_policy,
                discovery_address,
            } => {
                tracing::info!("Creating VllmPDRouter with prefill_urls: {:?}, decode_urls: {:?}, discovery: {:?}",
                              prefill_urls, decode_urls, discovery_address);
                Self::create_vllm_pd_router(
                    prefill_urls,
                    decode_urls,
                    discovery_address.clone(),
                    prefill_policy.as_ref(),
                    decode_policy.as_ref(),
                    &ctx.router_config.policy,
                    ctx,
                )
                .await
            }
            RoutingMode::OpenAI { worker_urls, .. } => {
                Self::create_openai_router(worker_urls.clone(), ctx).await
            }
        }
    }

    /// Create a regular router
    pub async fn create_regular_router(
        worker_urls: &[String],
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        if let Some(kv) = &ctx.router_config.kv_events {
            let policy = ctx.policy_registry.get_default_policy();
            let index = Arc::new(KVBlockIndex::new(kv.index_max_entries));
            let pool_config = KVEventPoolConfig {
                topic_filter: kv.topic_filter.clone(),
                default_kv_events_port: kv.default_port,
                endpoint_overrides: kv.endpoint_overrides.clone(),
            };
            let (mut pool, events) = KVEventPool::new(pool_config);
            for url in worker_urls {
                pool.subscribe_worker_by_http(url.clone(), url);
            }
            let tokenizer = create_tokenizer_async(&kv.tokenizer_path)
                .await
                .map_err(|error| format!("KV event tokenizer failed: {error}"))?
                as Arc<dyn Encoder>;
            let cache = Arc::new(EventCache { index: Arc::clone(&index), tokenizer, block_size: kv.block_size });
            if let Some(smetric) = policy.as_any().downcast_ref::<SMetricPolicy>() {
                smetric.enable_exact_cache(Arc::clone(&cache)).map_err(str::to_string)?;
            } else if let Some(cache_aware) = policy.as_any().downcast_ref::<CacheAwarePolicy>() {
                cache_aware.enable_exact_cache(cache).map_err(str::to_string)?;
            }
            tokio::spawn(run_kv_index_updater(events, index));
            tokio::spawn(async move {
                let _pool = pool;
                std::future::pending::<()>().await;
            });
            tracing::info!(
                workers = worker_urls.len(),
                block_size = kv.block_size,
                "PR130-derived KV events enabled; cache-aware policies use the common event index"
            );
        }
        // Create regular router with context
        let router = Router::new(worker_urls.to_vec(), ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a vLLM PD router with service discovery and/or static URLs
    pub async fn create_vllm_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        discovery_address: Option<String>,
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Initialize policies in PolicyRegistry - use specific policies if provided, otherwise fall back to main policy
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        // Set the prefill and decode policies in the registry
        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        // Create vLLM PD router with both static URLs and service discovery support
        if discovery_address.is_some() {
            tracing::info!(
                "Creating VllmPDRouter with service discovery at: {:?}",
                discovery_address
            );
        }
        if !prefill_urls.is_empty() || !decode_urls.is_empty() {
            tracing::info!(
                "Creating VllmPDRouter with static URLs - prefill: {:?}, decode: {:?}",
                prefill_urls,
                decode_urls
            );
        }

        let router = VllmPDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            discovery_address,
            ctx,
        )
        .await?;
        tracing::info!("VllmPDRouter instance created successfully");

        Ok(Box::new(router))
    }

    /// Create an OpenAI router
    async fn create_openai_router(
        worker_urls: Vec<String>,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Use the first worker URL as the OpenAI-compatible base
        let base_url = worker_urls
            .first()
            .cloned()
            .ok_or_else(|| "OpenAI mode requires at least one worker URL".to_string())?;

        let router =
            OpenAIRouter::new(base_url, Some(ctx.router_config.circuit_breaker.clone())).await?;

        Ok(Box::new(router))
    }
}
