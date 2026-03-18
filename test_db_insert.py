import asyncio
import json
import httpx
from content_engineer import ContentEngineer

async def main():
    print("Generating payload...")
    payload_str, provider, model, meta = ContentEngineer.generate_blog_payload(
        "Redis architecture", "google"
    )
    payload = json.loads(payload_str)
    print("Payload generated successfully.")
    
    print(f"Submitting to DB... Title: {payload.get('title')}")
    async with httpx.AsyncClient() as client:
        resp = await client.post("https://techxbytes.com/api/blogs", json=payload, timeout=20.0)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.text}")

if __name__ == "__main__":
    asyncio.run(main())
