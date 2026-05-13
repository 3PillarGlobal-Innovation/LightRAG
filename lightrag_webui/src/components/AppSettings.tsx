import { useState, useCallback, useEffect } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { useSettingsStore } from '@/stores/settings'
import { getWorkspaces, type WorkspaceInfo } from '@/api/lightrag'
import { PaletteIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'

const WORKSPACE_CUSTOM_SENTINEL = '__custom__'
const WORKSPACE_DEFAULT_SENTINEL = '__default__'

interface AppSettingsProps {
  className?: string
}

export default function AppSettings({ className }: AppSettingsProps) {
  const [opened, setOpened] = useState<boolean>(false)
  const { t } = useTranslation()

  const language = useSettingsStore.use.language()
  const setLanguage = useSettingsStore.use.setLanguage()

  const theme = useSettingsStore.use.theme()
  const setTheme = useSettingsStore.use.setTheme()

  const workspace = useSettingsStore.use.workspace()
  const setWorkspace = useSettingsStore.use.setWorkspace()
  const [knownWorkspaces, setKnownWorkspaces] = useState<WorkspaceInfo[]>([])
  const [workspaceMode, setWorkspaceMode] = useState<'select' | 'custom'>(
    workspace ? 'select' : 'select'
  )

  // Refresh workspace list whenever the popover opens. Cheap call; lets the
  // dropdown reflect any newly-indexed workspaces without a hard refresh.
  useEffect(() => {
    if (!opened) return
    let cancelled = false
    getWorkspaces()
      .then((list) => {
        if (cancelled) return
        setKnownWorkspaces(list)
        // If the currently-stored workspace isn't in the list, flip to
        // custom-input mode so the user sees their value.
        if (
          workspace &&
          !list.some((w) => w.workspace === workspace)
        ) {
          setWorkspaceMode('custom')
        }
      })
      .catch(() => {
        // /workspaces may not exist on older servers — silently fall back to
        // custom-text input so the feature still works.
        if (!cancelled) setWorkspaceMode('custom')
      })
    return () => {
      cancelled = true
    }
  }, [opened, workspace])

  const handleLanguageChange = useCallback((value: string) => {
    setLanguage(value as 'en' | 'zh' | 'fr' | 'ar' | 'zh_TW' | 'ru' | 'ja' | 'de' | 'uk' | 'ko' | 'vi')
  }, [setLanguage])

  const handleThemeChange = useCallback((value: string) => {
    setTheme(value as 'light' | 'dark' | 'system')
  }, [setTheme])

  const handleWorkspaceSelect = useCallback(
    (value: string) => {
      if (value === WORKSPACE_CUSTOM_SENTINEL) {
        setWorkspaceMode('custom')
        return
      }
      if (value === WORKSPACE_DEFAULT_SENTINEL) {
        setWorkspace('')
      } else {
        setWorkspace(value)
      }
      setWorkspaceMode('select')
      // Close the popover so the user sees the underlying view refresh.
      setOpened(false)
    },
    [setWorkspace]
  )

  const handleWorkspaceChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setWorkspace(e.target.value)
    },
    [setWorkspace]
  )

  const dropdownValue = (() => {
    if (workspaceMode === 'custom') return WORKSPACE_CUSTOM_SENTINEL
    if (!workspace) return WORKSPACE_DEFAULT_SENTINEL
    if (knownWorkspaces.some((w) => w.workspace === workspace)) return workspace
    // Stored workspace isn't (yet) in the list — render the dropdown blank
    // so the value doesn't visually clash with the available options.
    return undefined
  })()

  return (
    <Popover open={opened} onOpenChange={setOpened}>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="icon" className={cn('h-9 w-9', className)}>
          <PaletteIcon className="h-5 w-5" />
        </Button>
      </PopoverTrigger>
      <PopoverContent side="bottom" align="end" className="w-56">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">{t('settings.language')}</label>
            <Select value={language} onValueChange={handleLanguageChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="zh">中文</SelectItem>
                <SelectItem value="fr">Français</SelectItem>
                <SelectItem value="ar">العربية</SelectItem>
                <SelectItem value="zh_TW">繁體中文</SelectItem>
                <SelectItem value="ru">Русский</SelectItem>
                <SelectItem value="ja">日本語</SelectItem>
                <SelectItem value="de">Deutsch</SelectItem>
                <SelectItem value="uk">Українська</SelectItem>
                <SelectItem value="ko">한국어</SelectItem>
                <SelectItem value="vi">Tiếng Việt</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">{t('settings.theme')}</label>
            <Select value={theme} onValueChange={handleThemeChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="light">{t('settings.light')}</SelectItem>
                <SelectItem value="dark">{t('settings.dark')}</SelectItem>
                <SelectItem value="system">{t('settings.system')}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium">Workspace</label>
            <Select value={dropdownValue ?? ''} onValueChange={handleWorkspaceSelect}>
              <SelectTrigger>
                <SelectValue placeholder="Select a workspace…" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={WORKSPACE_DEFAULT_SENTINEL}>(server default)</SelectItem>
                {knownWorkspaces.map((w) => (
                  <SelectItem key={w.workspace} value={w.workspace || WORKSPACE_DEFAULT_SENTINEL}>
                    {w.workspace || '(empty)'}
                    {w.doc_count !== null ? ` — ${w.doc_count} docs` : ''}
                  </SelectItem>
                ))}
                <SelectItem value={WORKSPACE_CUSTOM_SENTINEL}>Custom…</SelectItem>
              </SelectContent>
            </Select>
            {workspaceMode === 'custom' && (
              <Input
                type="text"
                placeholder="solution_42"
                value={workspace}
                onChange={handleWorkspaceChange}
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
            )}
            <span className="text-xs text-muted-foreground">
              Sends as <code>LIGHTRAG-WORKSPACE</code> header. Blank = server default
              (e.g. <code>platform_docs</code>).
            </span>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}
