import win32api, win32con

user32 = WinDLL('user32', use_last_error=True)

HC_ACTION = 0
WH_MOUSE_LL = 14
VK_RETURN = 13
WM_QUIT        = 0x0012
WM_MOUSEMOVE   = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP   = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP   = 0x0205
WM_MBUTTONDOWN = 0x0207
WM_MBUTTONUP   = 0x0208
WM_MOUSEWHEEL  = 0x020A
WM_MOUSEHWHEEL = 0x020E

MSG_TEXT = {WM_MOUSEMOVE:   'WM_MOUSEMOVE',
            WM_LBUTTONDOWN: 'WM_LBUTTONDOWN',
            WM_LBUTTONUP:   'WM_LBUTTONUP',
            WM_RBUTTONDOWN: 'WM_RBUTTONDOWN',
            WM_RBUTTONUP:   'WM_RBUTTONUP',
            WM_MBUTTONDOWN: 'WM_MBUTTONDOWN',
            WM_MBUTTONUP:   'WM_MBUTTONUP',
            WM_MOUSEWHEEL:  'WM_MOUSEWHEEL',
            WM_MOUSEHWHEEL: 'WM_MOUSEHWHEEL'}

ULONG_PTR = WPARAM
LRESULT = LPARAM
LPMSG = POINTER(MSG)

HOOKPROC = WINFUNCTYPE(LRESULT, c_int, WPARAM, LPARAM)
LowLevelMouseProc = HOOKPROC

class MSLLHOOKSTRUCT(Structure):
    _fields_ = (('pt',          POINT),
                ('mouseData',   DWORD),
                ('flags',       DWORD),
                ('time',        DWORD),
                ('dwExtraInfo', ULONG_PTR))

LPMSLLHOOKSTRUCT = POINTER(MSLLHOOKSTRUCT)

user32.TranslateMessage.argtypes = (LPMSG,)
user32.DispatchMessageW.argtypes = (LPMSG,)

@LowLevelMouseProc
def LLMouseProc(nCode, wParam, lParam):
    msg = cast(lParam, LPMSLLHOOKSTRUCT)[0]
    msgid = MSG_TEXT.get(wParam, str(wParam))
    # msg = (msg.pt.x, msg.pt.y)
    msg = ((msg.pt.x, msg.pt.y),
            msg.mouseData, msg.flags,
            msg.time, msg.dwExtraInfo)   
    print('{:15s}: {}'.format(msgid, msg))
    return user32.CallNextHookEx(None, nCode, wParam, lParam)
    
def mouse_msg_loop():
    hHook = user32.SetWindowsHookExW(WH_MOUSE_LL, LLMouseProc, None, 0)
    msg = MSG()
    bRet = user32.GetMessageW(byref(msg), None, 0, 0)
    user32.TranslateMessage(byref(msg))
    user32.DispatchMessageW(byref(msg))
    
if __name__ == '__main__':
    t = threading.Thread(target=mouse_msg_loop)
    t.start()
