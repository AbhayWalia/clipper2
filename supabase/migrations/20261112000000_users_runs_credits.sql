-- Users, runs, clips and credits for the hosted clipper.
--
-- Files live in R2; these tables say who owns them. A run's files are under
--   clips/<user_id>/<run_id>/, recipes/<user_id>/<run_id>/ and vods/<user_id>/<run_id>.<ext>
-- so a run's id is that folder, "<user_id>/<run_id>": unique even when two users pick the
-- same run name.
--
-- Who writes what:
--   * Supabase Auth creates auth.users rows; a trigger adds the profile and the free credits.
--   * The server (clipper.py --upload, or the app's backend) writes runs, clips and credit
--     changes with the service-role key, which bypasses row level security.
--   * Signed-in users can only read their own rows. They never write directly, so they
--     can't give themselves credits or claim someone else's run.

-- --------------------------------------------------------------------------
-- Tables
-- --------------------------------------------------------------------------

create table public.profiles (
  id          uuid primary key references auth.users (id) on delete cascade,
  plan        text not null default 'free' check (plan in ('free', 'paid')),
  created_at  timestamptz not null default now()
);

create table public.runs (
  id              text primary key,          -- "<user_id>/<run_id>", the run's folder in R2
  user_id         uuid not null references public.profiles (id) on delete cascade,
  video_name      text,
  video_minutes   numeric(8, 2),
  mode            text check (mode in ('copy', 'encode', 'preview')),
  status          text not null default 'uploaded'
                  check (status in ('processing', 'uploaded', 'failed', 'deleted')),
  clip_count      int not null default 0,
  cost_usd        numeric(10, 4),            -- Claude cost recorded in clips.json
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  check (id like user_id::text || '/%')    -- a run can only sit in its owner's folder
);
create index runs_user_created_idx on public.runs (user_id, created_at desc);

create table public.clips (
  run_id       text not null references public.runs (id) on delete cascade,
  idx          int not null,                 -- the clip's number, as in 01_title.mp4
  title        text,
  hook         text,
  category     text,
  score        numeric(4, 2),
  start_s      numeric(10, 3),               -- content range in the source video
  end_s        numeric(10, 3),
  file_key     text,                         -- latest render's R2 key (full quality or preview)
  is_preview   boolean not null default false,
  size_bytes   bigint,
  review       text check (review in ('approved', 'rejected')),  -- as the web UI saves it
  updated_at   timestamptz not null default now(),
  primary key (run_id, idx)
);

-- One row per change, never edited: the balance is the sum, so every credit can be traced.
create table public.credit_ledger (
  id                 bigint generated always as identity primary key,
  user_id            uuid not null references public.profiles (id) on delete cascade,
  delta              int not null check (delta <> 0),
  reason             text not null
                     check (reason in ('free_signup', 'purchase', 'render', 'refund', 'adjustment')),
  run_id             text references public.runs (id) on delete set null,
  stripe_payment_id  text unique,            -- a payment can only be credited once
  created_at         timestamptz not null default now()
);
create index credit_ledger_user_idx on public.credit_ledger (user_id);

create view public.credit_balance with (security_invoker = true) as
  select user_id, sum(delta)::int as balance
  from public.credit_ledger
  group by user_id;

-- --------------------------------------------------------------------------
-- New users: a profile and the free clips
-- --------------------------------------------------------------------------

create function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  insert into public.profiles (id) values (new.id);
  insert into public.credit_ledger (user_id, delta, reason) values (new.id, 6, 'free_signup');
  return new;
end;
$$;

create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- --------------------------------------------------------------------------
-- Spending credits: check and deduct in one step, so two renders started at
-- the same moment can't both spend the last credit.
-- --------------------------------------------------------------------------

create function public.spend_credits(p_user uuid, p_amount int, p_run text)
returns int
language plpgsql security definer set search_path = '' as $$
declare
  bal int;
begin
  if p_amount <= 0 then
    raise exception 'amount must be positive';
  end if;
  perform pg_advisory_xact_lock(hashtextextended(p_user::text, 0));
  select coalesce(sum(delta), 0) into bal from public.credit_ledger where user_id = p_user;
  if bal < p_amount then
    raise exception 'not enough credits: % left, % needed', bal, p_amount
      using errcode = 'P0001';
  end if;
  insert into public.credit_ledger (user_id, delta, reason, run_id)
    values (p_user, -p_amount, 'render', p_run);
  return bal - p_amount;
end;
$$;

-- Only the server may run these; signed-in users must not spend or grant credits themselves.
revoke execute on function public.spend_credits(uuid, int, text) from public, anon, authenticated;
grant execute on function public.spend_credits(uuid, int, text) to service_role;
revoke execute on function public.handle_new_user() from public, anon, authenticated;

-- --------------------------------------------------------------------------
-- Row level security: users read their own rows; only the server writes.
-- --------------------------------------------------------------------------

alter table public.profiles      enable row level security;
alter table public.runs          enable row level security;
alter table public.clips         enable row level security;
alter table public.credit_ledger enable row level security;

create policy "read own profile" on public.profiles
  for select to authenticated using ((select auth.uid()) = id);

create policy "read own runs" on public.runs
  for select to authenticated using ((select auth.uid()) = user_id);

create policy "read own clips" on public.clips
  for select to authenticated using (
    exists (select 1 from public.runs r
            where r.id = clips.run_id and r.user_id = (select auth.uid())));

create policy "read own credits" on public.credit_ledger
  for select to authenticated using ((select auth.uid()) = user_id);
