//! Resolve the `source` form field (image URL or base64) into image bytes,
//! with SSRF protection for URL fetches.
//!
//! SSRF model (when `allow_private_urls` is false):
//! - every hostname the HTTP client connects to — the initial URL *and* every
//!   redirect hop — goes through [`PublicOnlyResolver`], so the address that
//!   is validated is the address that is dialed (no DNS-rebinding window);
//! - IP-literal hosts never hit a resolver, so they are checked explicitly on
//!   the initial URL and in the redirect policy;
//! - proxies from the environment are disabled (they would bypass both).

use std::error::Error as StdError;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD};
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::config::Limits;
use crate::error::{ApiError, Result};

const MAX_REDIRECTS: usize = 5;

pub async fn resolve_source(
    client: &reqwest::Client,
    source: &str,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if let Some(rest) = source.strip_prefix("data:") {
        let (_, b64) = rest
            .split_once(',')
            .ok_or_else(|| ApiError::BadRequest("malformed data URI in source".into()))?;
        return decode_base64(b64);
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        fetch_url(client, source, limits).await
    } else {
        decode_base64(source)
    }
}

fn decode_base64(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    STANDARD
        .decode(s)
        .or_else(|_| URL_SAFE_NO_PAD.decode(s))
        .map_err(|_| ApiError::BadRequest("source is not a valid base64-encoded image".into()))
}

/// Build the HTTP client used for `source` URLs.
pub fn fetch_client(limits: &Limits) -> reqwest::Result<reqwest::Client> {
    let builder = reqwest::Client::builder()
        .timeout(Duration::from_millis(limits.url_timeout_ms))
        .no_proxy();
    if limits.allow_private_urls {
        return builder
            .redirect(reqwest::redirect::Policy::limited(MAX_REDIRECTS))
            .build();
    }
    builder
        .dns_resolver(PublicOnlyResolver)
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= MAX_REDIRECTS {
                attempt.error("too many redirects")
            } else if literal_ip(attempt.url()).is_some_and(is_forbidden) {
                attempt.error(Blocked)
            } else {
                attempt.follow()
            }
        }))
        .build()
}

async fn fetch_url(client: &reqwest::Client, url: &str, limits: &Limits) -> Result<Vec<u8>> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|_| ApiError::BadRequest("source is not a valid URL".into()))?;
    if parsed.host_str().is_none() {
        return Err(ApiError::BadRequest("source URL has no host".into()));
    }
    if !limits.allow_private_urls && literal_ip(&parsed).is_some_and(is_forbidden) {
        return Err(blocked_error());
    }

    let mut resp = client.get(parsed).send().await.map_err(|e| {
        if is_blocked(&e) {
            blocked_error()
        } else {
            ApiError::BadRequest(format!("failed to fetch source URL: {e}"))
        }
    })?;
    if !resp.status().is_success() {
        return Err(ApiError::BadRequest(format!(
            "source URL returned status {}",
            resp.status().as_u16()
        )));
    }

    let max = limits.max_upload_mb.saturating_mul(1024 * 1024);
    let too_large = || ApiError::TooLarge("source image too large".into());
    if resp.content_length().is_some_and(|len| len > max) {
        return Err(too_large());
    }
    // Stream with a running cap: Content-Length may be absent (chunked).
    let mut body = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| ApiError::BadRequest(format!("failed to read source URL body: {e}")))?
    {
        if (body.len() + chunk.len()) as u64 > max {
            return Err(too_large());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn blocked_error() -> ApiError {
    ApiError::BadRequest("source URL resolves to a private or reserved address".into())
}

/// Marker error raised by the resolver / redirect policy.
#[derive(Debug)]
struct Blocked;

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("destination address is private or reserved")
    }
}

impl StdError for Blocked {}

fn is_blocked(e: &reqwest::Error) -> bool {
    let mut cur: Option<&(dyn StdError + 'static)> = Some(e);
    while let Some(err) = cur {
        if err.is::<Blocked>() {
            return true;
        }
        cur = err.source();
    }
    false
}

/// DNS resolver that refuses names resolving to any non-public address.
struct PublicOnlyResolver;

impl Resolve for PublicOnlyResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let host = name.as_str().to_string();
        Box::pin(async move {
            let addrs: Vec<SocketAddr> =
                tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
            if addrs.is_empty() {
                return Err(format!("{host} did not resolve").into());
            }
            if addrs.iter().any(|a| is_forbidden(a.ip())) {
                return Err(Box::new(Blocked) as Box<dyn StdError + Send + Sync>);
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

fn literal_ip(url: &reqwest::Url) -> Option<IpAddr> {
    url.host_str()?
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .ok()
}

/// True for any address that is not globally routable unicast.
pub fn is_forbidden(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => forbidden_v4(v4),
        IpAddr::V6(v6) => forbidden_v6(v6),
    }
}

fn forbidden_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_private()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_multicast()
        || ip.is_documentation()
        || o[0] == 0 // 0.0.0.0/8 "this network"
        || (o[0] == 100 && (o[1] & 0xc0) == 64) // 100.64.0.0/10 shared / CGNAT
        || (o[0] == 192 && o[1] == 0 && o[2] == 0) // 192.0.0.0/24 IETF protocol
        || (o[0] == 198 && (o[1] & 0xfe) == 18) // 198.18.0.0/15 benchmarking
        || o[0] >= 240 // 240.0.0.0/4 reserved
}

fn forbidden_v6(ip: Ipv6Addr) -> bool {
    let s = ip.segments();
    // Embedded IPv4: mapped (::ffff:a.b.c.d), compatible (::a.b.c.d),
    // NAT64 (64:ff9b::/96) and 6to4 (2002::/16) all reach an IPv4 host.
    if let Some(v4) = ip.to_ipv4_mapped() {
        return forbidden_v4(v4);
    }
    if s[..6] == [0, 0, 0, 0, 0, 0] && !ip.is_unspecified() && !ip.is_loopback() {
        return forbidden_v4(Ipv4Addr::from(((s[6] as u32) << 16) | s[7] as u32));
    }
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        return forbidden_v4(Ipv4Addr::from(((s[6] as u32) << 16) | s[7] as u32));
    }
    if s[0] == 0x2002 {
        return forbidden_v4(Ipv4Addr::from(((s[1] as u32) << 16) | s[2] as u32));
    }
    ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || (s[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
        || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link local
        || (s[0] & 0xffc0) == 0xfec0 // fec0::/10 site local (deprecated)
        || (s[0] == 0x2001 && s[1] == 0x0db8) // 2001:db8::/32 documentation
        || (s[0] == 0x2001 && s[1] == 0) // 2001::/32 Teredo
        || (s[0] == 0x0100 && s[1..4] == [0, 0, 0]) // 100::/64 discard
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(s: &str) -> bool {
        is_forbidden(s.parse().unwrap())
    }

    #[test]
    fn blocks_private_and_reserved() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "0.0.0.0",
            "0.1.2.3",
            "100.64.0.1",
            "198.18.0.1",
            "224.0.0.1",
            "255.255.255.255",
            "240.0.0.1",
            "::1",
            "::",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
            "::127.0.0.1",
            "64:ff9b::a9fe:a9fe",
            "2002:7f00:1::",
            "2001:db8::1",
        ] {
            assert!(f(ip), "{ip} must be blocked");
        }
    }

    #[test]
    fn allows_public() {
        for ip in [
            "8.8.8.8",
            "1.1.1.1",
            "100.128.0.1",
            "2606:4700:4700::1111",
            "::ffff:8.8.8.8",
        ] {
            assert!(!f(ip), "{ip} must be allowed");
        }
    }

    #[test]
    fn literal_ip_parses_v6_brackets() {
        let u = reqwest::Url::parse("http://[::1]:8080/x").unwrap();
        assert_eq!(literal_ip(&u), Some("::1".parse().unwrap()));
        let u = reqwest::Url::parse("http://example.com/x").unwrap();
        assert_eq!(literal_ip(&u), None);
    }
}
