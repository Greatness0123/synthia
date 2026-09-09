/**
 * Central icon re-exports using @material-symbols-svg/react (Outlined variant).
 * All icons are tree-shaken at build time via Vite optimizeDeps.
 * Usage: import { Brain, Trash, Eye } from '../ui/icons';
 */
import type { IconProps as MaterialIconProps } from '@material-symbols-svg/react';

export type IconProps = MaterialIconProps;

export {
  // Core UI
  Close as X,
  Visibility as Eye,
  Info,
  Search,
  Search as MagnifyingGlass,
  Settings,
  Settings as GearSix,
  Settings as Gear,
  Tune as SlidersHorizontal,

  // Navigation / Actions
  ArrowForward as ArrowRight,
  ArrowDownward as ArrowDown,
  ArrowUpward as ArrowUp,
  Upload as ArrowFatLinesUp,
  Refresh as ArrowsClockwise,
  Download as DownloadSimple,
  Upload as UploadSimple,
  Download as Export,
  ContentCopy as Copy,

  // Status / Feedback
  Check,
  CheckCircle,
  CheckCircle as CheckCircleFilled,
  Warning as WarningCircle,
  Warning as WarningFilled,
  Cancel as XCircle,
  Info as InfoFilled,
  ProgressActivity as Spinner,
  Wifi as WifiHigh,
  ElectricalServices as PlugsConnected,

  // Agent / AI
  Psychology as Brain,
  SmartToy as Robot,
  Memory as Cpu,
  Orthopedics as Bone,
  Vaccines as Syringe,
  Biotech as Dna,

  // Media / Sensors
  PhotoCamera as Camera,
  Videocam as VideoCamera,
  Monitor,
  LightMode as Sun,
  DarkMode as Moon,
  VolumeUp as SpeakerHigh,
  VolumeOff as SpeakerSlash,
  Mic as Microphone,
  Stop as StopSquare,
  MusicNote as MusicNotes,
  MusicNote as MusicNote,

  // Data / Files
  Database,
  Archive,
  Code as FileCode,
  TableChart as FileCsv,
  Description as Document,
  MenuBook as Notebook,
  Bookmark,
  CloudUpload as FileCloud,

  // 3D / World
  DeployedCode as Cube,
  Circle,
  ChangeHistory as Triangle,
  DataObject as Cylinder,
  GridView as DotsNine,
  GridOn as Grid,

  // People / Social
  Person as User,
  Group as People,

  // Section / Misc
  Public as Globe,
  Tune as Sliders,
  SmartToy as Bot,
  Bolt as Zap,
  Sync as RefreshCw,
  Flag,
  MyLocation as Target,
  KeyboardArrowDown as CaretDown,
  KeyboardArrowUp as CaretUp,
  Checklist as ListChecks,
  AccountTree as TreeStructure,
  TrendingUp as TrendUp,
  ShowChart as ArrowTrendingLines,
  DirectionsWalk as Steps,
  OpenWith as Move,
  Delete as Trash,
  PlayArrow as Play,
  Pause,
  Save,

} from '@material-symbols-svg/react';
