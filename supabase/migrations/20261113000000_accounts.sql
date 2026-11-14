-- Account details and linked YouTube / Twitch accounts.
--
-- Run after 20261112000000_users_runs_credits.sql. As there, signed-in users only read
-- their own rows and the server writes with the service-role key.

-- --------------------------------------------------------------------------
-- Profile fields the user edits on the account page
-- --------------------------------------------------------------------------

alter table public.profiles
  add column display_name text check (char_length(display_name) between 1 and 50),
  add column updated_at   timestamptz not null default now();

-- --------------------------------------------------------------------------
-- Linked accounts: one row per YouTube channel or Twitch user a user linked
-- --------------------------------------------------------------------------

create table public.linked_accounts (
  id                bigint generated always as identity primary key,
  user_id           uuid not null references public.profiles (id) on delete cascade,
  provider          text not null check (provider in ('youtube', 'twitch')),
  provider_user_id  text not null,           -- YouTube channel id / Twitch user id
  handle            text,                    -- YouTube @handle / Twitch login
  display_name      text,
  avatar_url        text,
  scopes            text[] not null default '{}',
  -- OAuth tokens: the server reads them, signed-in users can't (see the grants below).
  -- TODO: encrypt with Supabase Vault (or an app key) before real users link accounts.
  access_token      text,
  refresh_token     text,
  token_expires_at  timestamptz,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now(),
  unique (provider, provider_user_id)        -- a channel belongs to one clipper account
);
create index linked_accounts_user_idx on public.linked_accounts (user_id);

alter table public.linked_accounts enable row level security;

create policy "read own linked accounts" on public.linked_accounts
  for select to authenticated using ((select auth.uid()) = user_id);

-- Supabase grants every table to anon and authenticated by default. Take that back and
-- give signed-in users every column except the tokens.
revoke all on public.linked_accounts from anon, authenticated;
grant select (id, user_id, provider, provider_user_id, handle, display_name, avatar_url,
              scopes, token_expires_at, created_at, updated_at)
  on public.linked_accounts to authenticated;
